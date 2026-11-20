#!/usr/bin/env python3
"""camera_calib_node.py - estimates the camera pose (TF world -> camera optical frame).

The subject: "on the real robot, nobody hands you the camera extrinsics". An A4 sheet with
4 ArUco markers (sim/.../calib/make_calib_sheet.py) is laid at a known place in the robot
frame: parameter `sheet_poses` = [x, y, z, yaw_deg] of the sheet centre (sheet frame: x
along the short side, y along the long side, drawn on the sheet). The Ouija board (its 4
corner markers, ouija_layout.json) can be used the same way: `sheet_files` lists the
layouts, `sheet_poses` their poses (4 numbers each). The marker corners are expressed in
the world frame, so solvePnP on all the detected corners gives T_camera_world directly,
whatever the sheets seen (ids are distinct: 0-3 board, 10-13 sheet).
`num_samples` estimates are averaged, the static TF is published (replacing the ground
truth of the simulator, see `publish_camera_tf` in the driver) and, when it publishes the
TF (real camera), saved to `calib_file` so the sheet can be removed afterwards: with
`mode: load` the saved TF is re-published. A simulation run never overwrites that file.

In simulation the driver still publishes the ground truth as `external_cam_optical_gt`
(`gt_frame`): the node logs the error of its estimate against it.

Topics: /camera_calib/debug_image, /camera_calib/status (std_msgs/String).
"""
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import rclpy.node
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

DEFAULT_SHEET = "/opt/so101/sim/so101_sim/assets/so101/calib/calib_sheet.json"
DEFAULT_CALIB_FILE = "/ros2_ws/calibration/camera_tf.json"


def _quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_quat(R):
    """Rotation matrix -> (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s, 0.25 / s
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


def _average_transforms(Ts):
    """Mean translation, mean rotation (quaternion average, fine for close rotations)."""
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    qs = np.array([_matrix_to_quat(T[:3, :3]) for T in Ts])
    qs[np.dot(qs, qs[0]) < 0] *= -1  # same hemisphere
    q = np.mean(qs, axis=0)
    q /= np.linalg.norm(q)
    T = np.eye(4)
    T[:3, :3] = _quat_to_matrix(*q)
    T[:3, 3] = t
    return T


def _pose_error(T_a, T_b):
    """(translation error m, rotation error deg) between two 4x4 poses."""
    dt = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    R = T_a[:3, :3].T @ T_b[:3, :3]
    c = min(1.0, max(-1.0, (np.trace(R) - 1) / 2))
    return dt, math.degrees(math.acos(c))


class CameraCalibNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("camera_calib")
        p = self.declare_parameter
        self._camera_topic = p("camera_topic", "/external_cam").value
        self._world_frame = p("world_frame", "world").value
        self._camera_frame = p("camera_frame", "").value  # empty: frame_id of camera_info
        sheet_files = list(p("sheet_files", [DEFAULT_SHEET]).value)
        # Sheet centres in the robot frame: x, y, z (m) and yaw (deg) of the sheet x axis, 4 per sheet.
        self._sheet_poses = [float(v) for v in p("sheet_poses", [0.0, 0.0, 0.0, 0.0]).value]
        self._num_samples = int(p("num_samples", 30).value)
        # Measured in sim (Ouija board): 2 markers -> 15-22 mm on the camera position, 3 -> 7 mm.
        # (In the Ouija scene marker 0 is out of the camera view: 3 is the maximum there.)
        self._min_markers = int(p("min_markers", 3).value)
        self._mode = p("mode", "calibrate").value  # calibrate | load
        self._calib_file = p("calib_file", DEFAULT_CALIB_FILE).value
        self._publish_tf = bool(p("publish_tf", True).value)
        self._gt_frame = p("gt_frame", "external_cam_optical_gt").value

        if len(self._sheet_poses) != 4 * len(sheet_files):
            raise ValueError(f"sheet_poses needs 4 values per sheet ({len(sheet_files)} sheets)")
        self._obj_world = {}  # marker id -> 4x3 corners in the world frame (detectMarkers order)
        self._sheets = []
        for k, sheet_file in enumerate(sheet_files):
            layout = json.loads(Path(sheet_file).read_text())
            x, y, z, yaw = self._sheet_poses[4 * k:4 * k + 4]
            c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            T_ws = np.array([[c, -s_, 0, x], [s_, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)
            size = float(layout["marker_size"])
            u_dir, v_dir = (np.array(a, dtype=float) for a in layout.get("marker_axes", [[1, 0], [0, -1]]))
            for mid, centre in layout["markers"].items():
                cxy = np.array(centre, dtype=float)
                h = size / 2
                # top-left, top-right, bottom-right, bottom-left of the marker image
                corners = [cxy - h * u_dir - h * v_dir, cxy + h * u_dir - h * v_dir,
                           cxy + h * u_dir + h * v_dir, cxy - h * u_dir + h * v_dir]
                self._obj_world[int(mid)] = np.array(
                    [(T_ws @ np.array([cx, cy, 0.0, 1.0]))[:3] for cx, cy in corners])
            self._sheets.append(f"{Path(sheet_file).name} at ({x}, {y}, {z}, {yaw} deg), ids {sorted(layout['markers'])}")
        # Sub-pixel corners matter here: the markers are small in the image and the camera is
        # far, so a 1 px error is ~1 cm on the camera position (sim: 11 mm -> 5 mm with refinement).
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self._detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)

        self._bridge = CvBridge()
        self._K = None
        self._samples = []
        self._sample_markers = []
        self._T_wc = None  # final estimate
        self._done = False
        self._t_last_log = 0.0
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_static = StaticTransformBroadcaster(self)

        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, f"{self._camera_topic}/camera_info", self._cb_camera_info, 5)
        # Best Effort, like the camera publisher (sim driver or real camera): see README "QoS".
        self.create_subscription(Image, f"{self._camera_topic}/image_raw", self._cb_image, qos_profile_sensor_data)
        self._pub_debug = self.create_publisher(Image, "/camera_calib/debug_image", qos)
        self._pub_status = self.create_publisher(String, "/camera_calib/status", 5)
        self.create_timer(1.0, self._publish_status)

        if self._mode == "load":
            self._load()
        self.get_logger().info(f"Camera calib node ready: mode={self._mode}, {self._num_samples} samples, "
                               f"sheets: {'; '.join(self._sheets)}")

    # ----------------------------------------------------------------- inputs

    def _cb_camera_info(self, msg: CameraInfo):
        if msg.k[0] <= 0.0:  # real camera without intrinsics file: K is all zeros
            self.get_logger().warning("camera_info without intrinsics (fx = 0): calibrate the camera "
                                      "(ros2_ws/calibration/camera_info.yaml)", throttle_duration_sec=10.0)
            return
        self._K = np.array(msg.k, dtype=float).reshape(3, 3)
        if not self._camera_frame:
            self._camera_frame = msg.header.frame_id

    def _cb_image(self, msg: Image):
        if self._done or self._mode == "load":
            return
        if self._K is None or not self._camera_frame:
            self.get_logger().info("waiting for camera_info...", throttle_duration_sec=5.0)
            return
        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        est = self._estimate(bgr)
        if est is None:
            self.get_logger().info("calibration sheet not visible (need >= %d markers)" % self._min_markers,
                                   throttle_duration_sec=5.0)
            self._publish_debug(bgr, None, msg)
            return
        T_cw, used, corners, ids, reproj = est
        self._samples.append(np.linalg.inv(T_cw))
        self._sample_markers.append(len(used))
        self._publish_debug(bgr, (corners, ids, T_cw, reproj), msg)
        if len(self._samples) >= self._num_samples:
            self._finish()

    # ------------------------------------------------------------------ PnP

    def _estimate(self, bgr):
        corners, ids, _ = self._detector.detectMarkers(bgr)
        if ids is None:
            return None
        obj, img, used = [], [], []
        for c, i in zip(corners, ids.flatten()):
            if int(i) in self._obj_world:
                obj.append(self._obj_world[int(i)])
                img.append(c[0])
                used.append(int(i))
        if len(used) < self._min_markers:
            return None
        obj = np.concatenate(obj).astype(np.float64)
        img = np.concatenate(img).astype(np.float64)
        # SQPnP: no planarity assumption (two sheets at different heights may be seen).
        ok, rvec, tvec = cv2.solvePnP(obj, img, self._K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, self._K, None, rvec, tvec)
        T_cw = np.eye(4)  # world expressed in the camera frame
        T_cw[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cw[:3, 3] = tvec.flatten()
        reproj = cv2.projectPoints(obj, rvec, tvec, self._K, None)[0].reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum((reproj - img) ** 2, axis=1))))
        return T_cw, used, corners, ids, err

    # --------------------------------------------------------------- results

    def _finish(self):
        self._T_wc = _average_transforms(self._samples)
        spread = np.max([np.linalg.norm(T[:3, 3] - self._T_wc[:3, 3]) for T in self._samples])
        self._done = True
        t = self._T_wc[:3, 3]
        self.get_logger().info(
            f"camera pose estimated from {len(self._samples)} samples ({min(self._sample_markers)}-"
            f"{max(self._sample_markers)} markers): t=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m, "
            f"sample spread {spread * 1000:.1f} mm"
        )
        self._compare_with_ground_truth()
        if self._publish_tf:
            self._send_tf()
            self._save()
        else:
            # camera_tf:=sim: evaluation against the simulator only; the saved calibration of the
            # real camera (calibration/camera_tf.json, versioned) must not be overwritten by a sim run
            self.get_logger().info(f"evaluation only (publish_tf=false): {self._calib_file} left untouched")

    def _compare_with_ground_truth(self):
        try:
            tf = self._tf_buffer.lookup_transform(self._world_frame, self._gt_frame, rclpy.time.Time())
        except Exception:  # noqa: BLE001
            self.get_logger().info(f"no ground truth frame '{self._gt_frame}' (real camera): no comparison")
            return
        t, q = tf.transform.translation, tf.transform.rotation
        T_gt = np.eye(4)
        T_gt[:3, :3] = _quat_to_matrix(q.x, q.y, q.z, q.w)
        T_gt[:3, 3] = [t.x, t.y, t.z]
        dt, dr = _pose_error(self._T_wc, T_gt)
        self.get_logger().info(
            f"vs ground truth: position error {dt * 1000:.1f} mm, orientation error {dr:.2f} deg "
            f"(gt t=({t.x:.4f}, {t.y:.4f}, {t.z:.4f}))"
        )

    def _send_tf(self):
        T = self._T_wc
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._world_frame
        msg.child_frame_id = self._camera_frame
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, T[:3, 3])
        qx, qy, qz, qw = _matrix_to_quat(T[:3, :3])
        msg.transform.rotation.x, msg.transform.rotation.y = float(qx), float(qy)
        msg.transform.rotation.z, msg.transform.rotation.w = float(qz), float(qw)
        self._tf_static.sendTransform(msg)
        self.get_logger().info(f"static TF {self._world_frame} -> {self._camera_frame} published")

    def _save(self):
        if not self._calib_file:
            return
        qx, qy, qz, qw = _matrix_to_quat(self._T_wc[:3, :3])
        data = {"world_frame": self._world_frame, "camera_frame": self._camera_frame,
                "translation": [float(v) for v in self._T_wc[:3, 3]],
                "rotation_xyzw": [float(qx), float(qy), float(qz), float(qw)],
                "sheets": self._sheets, "num_samples": len(self._samples),
                "date": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            path = Path(self._calib_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=1))
            self.get_logger().info(f"saved to {self._calib_file}")
        except OSError as exc:
            self.get_logger().warning(f"could not save {self._calib_file}: {exc}")

    def _load(self):
        try:
            data = json.loads(Path(self._calib_file).read_text())
        except (OSError, ValueError) as exc:
            self.get_logger().error(f"mode=load but cannot read {self._calib_file}: {exc}")
            return
        self._T_wc = np.eye(4)
        self._T_wc[:3, :3] = _quat_to_matrix(*data["rotation_xyzw"])
        self._T_wc[:3, 3] = data["translation"]
        self._camera_frame = self._camera_frame or data["camera_frame"]
        self._done = True
        self.get_logger().info(f"loaded camera pose from {self._calib_file} ({data.get('date', '?')})")
        if self._publish_tf:
            self._send_tf()

    # ------------------------------------------------------------------ debug

    def _publish_status(self):
        if self._done:
            txt = "done" if self._T_wc is not None else "failed"
        else:
            txt = f"sampling {len(self._samples)}/{self._num_samples}"
        self._pub_status.publish(String(data=txt))

    def _publish_debug(self, bgr, est, msg):
        if self._pub_debug.get_subscription_count() == 0:
            return
        dbg = bgr.copy()
        if est is not None:
            corners, ids, T_cw, reproj = est
            cv2.aruco.drawDetectedMarkers(dbg, corners, ids)
            rvec = cv2.Rodrigues(T_cw[:3, :3])[0]
            cv2.drawFrameAxes(dbg, self._K, None, rvec, T_cw[:3, 3], 0.1)  # world frame (robot base)
            txt = f"{len(ids)} markers, reproj {reproj:.2f} px, sample {len(self._samples)}/{self._num_samples}"
        else:
            txt = "calibration sheet not visible"
        cv2.putText(dbg, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        out = self._bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
        out.header = msg.header
        self._pub_debug.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
