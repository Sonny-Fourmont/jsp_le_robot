#!/usr/bin/env python3
"""ouija_perception_node.py - locates the Ouija board and the planchette in 3D.

  1. Board: the 4 ArUco markers (DICT_4X4_50, ids 0..3) printed in the board corners are
     detected, and solvePnP with their 16 corners (positions known from ouija_layout.json)
     gives the board pose in the camera frame. With the camera TF (world -> optical frame)
     the board pose is expressed in the world: /ouija/board_pose. The letters are then
     known in the world (layout targets are in the board frame): /ouija/targets_markers.
  2. Planchette (a blue ring): blue blob inside the board outline -> centroid of its outer
     contour -> ray/plane intersection at the ring's mid height, like the phase 1
     perception: /ouija/planchette_position,
     plus /ouija/pointed (std_msgs/String): the nearest target label, or "" if the puck is
     not detected / is partly hidden by the arm.
Debug image with everything drawn on /ouija/debug_image.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import rclpy
import rclpy.node
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

DEFAULT_LAYOUT = "/opt/so101/sim/so101_sim/assets/so101/ouija/ouija_layout.json"
BOARD_THICKNESS = 0.004  # m, markers are printed on the top face


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


class OuijaPerceptionNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("ouija_perception")
        p = self.declare_parameter
        self._camera_topic = p("camera_topic", "/external_cam").value
        self._world_frame = p("world_frame", "world").value
        layout_file = p("layout_file", DEFAULT_LAYOUT).value
        self._puck_lo = np.array(p("planchette_hsv_low", [100, 120, 80]).value, dtype=np.uint8)
        self._puck_hi = np.array(p("planchette_hsv_high", [130, 255, 255]).value, dtype=np.uint8)
        self._min_visible_ratio = p("min_visible_ratio", 0.6).value  # blob area / expected area
        self._pointed_tolerance = p("pointed_tolerance", 0.02).value  # m

        layout = json.loads(Path(layout_file).read_text())
        self._marker_size = float(layout["marker_size"])
        self._markers = {int(k): np.array(v, dtype=float) for k, v in layout["markers"].items()}
        self._targets = {k: np.array(v, dtype=float) for k, v in layout["targets"].items()}
        self._board_size = layout["board_size"]  # [x, y]
        self._puck_radius = float(layout.get("planchette_radius", 0.021))
        self._puck_height = float(layout.get("planchette_height", 0.015))
        self.get_logger().info(f"layout: {len(self._targets)} targets, {len(self._markers)} markers")

        self._bridge = CvBridge()
        self._K = None
        self._camera_frame = None
        self._T_wc = None
        self._T_wb = None  # world <- board (last estimate)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, f"{self._camera_topic}/camera_info", self._cb_camera_info, 5)
        self.create_subscription(Image, f"{self._camera_topic}/image_raw", self._cb_image, 2)
        self._pub_board = self.create_publisher(PoseStamped, "/ouija/board_pose", 5)
        self._pub_puck = self.create_publisher(PoseStamped, "/ouija/planchette_position", 5)
        self._pub_pointed = self.create_publisher(String, "/ouija/pointed", 5)
        self._pub_markers = self.create_publisher(MarkerArray, "/ouija/targets_markers", 2)
        self._pub_debug = self.create_publisher(Image, "/ouija/debug_image", qos)
        self._n_frames = 0
        self.get_logger().info("Ouija perception ready.")

    # ----------------------------------------------------------------- inputs

    def _cb_camera_info(self, msg: CameraInfo):
        if msg.k[0] <= 0.0:  # real camera without intrinsics file: K is all zeros
            self.get_logger().warning("camera_info without intrinsics (fx = 0): calibrate the camera "
                                      "(ros2_ws/calibration/camera_info.yaml)", throttle_duration_sec=10.0)
            return
        self._K = np.array(msg.k, dtype=float).reshape(3, 3)
        self._camera_frame = msg.header.frame_id

    def _fetch_cam_params(self) -> bool:
        if self._K is None or not self._camera_frame:
            return False
        if self._T_wc is None:
            try:
                tf = self._tf_buffer.lookup_transform(self._world_frame, self._camera_frame, rclpy.time.Time())
            except Exception:  # noqa: BLE001
                return False
            t, q = tf.transform.translation, tf.transform.rotation
            self._T_wc = np.eye(4)
            self._T_wc[:3, :3] = _quat_to_matrix(q.x, q.y, q.z, q.w)
            self._T_wc[:3, 3] = [t.x, t.y, t.z]
            self.get_logger().info(f"camera pose received: t={np.round(self._T_wc[:3, 3], 3)}")
        return True

    # ------------------------------------------------------------------ board

    def _marker_object_points(self, mid):
        """Corners of marker `mid` in the board frame, in the detectMarkers order
        (top-left, top-right, bottom-right, bottom-left of the printed marker image).
        The marker image is printed with its u axis along -y_board and v along -x_board."""
        bx, by = self._markers[mid]
        h = self._marker_size / 2
        return np.array([[bx + h, by + h, 0], [bx + h, by - h, 0], [bx - h, by - h, 0], [bx - h, by + h, 0]])

    def _estimate_board(self, bgr):
        """ArUco + PnP -> (T_cb, ids, corners) or None."""
        corners, ids, _ = self._detector.detectMarkers(bgr)
        if ids is None:
            return None
        obj, img, used = [], [], []
        for c, i in zip(corners, ids.flatten()):
            if int(i) in self._markers:
                obj.append(self._marker_object_points(int(i)))
                img.append(c[0])
                used.append(int(i))
        if len(used) < 2:  # PnP on a single marker is ambiguous
            return None
        obj = np.concatenate(obj).astype(np.float64)
        img = np.concatenate(img).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, self._K, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, self._K, None, rvec, tvec)
        T_cb = np.eye(4)
        T_cb[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cb[:3, 3] = tvec.flatten()
        reproj = cv2.projectPoints(obj, rvec, tvec, self._K, None)[0].reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum((reproj - img) ** 2, axis=1))))
        return T_cb, used, corners, ids, err

    def _board_to_world(self, bxy, z=0.0):
        return (self._T_wb @ np.array([bxy[0], bxy[1], z, 1.0]))[:3]

    # ------------------------------------------------------------- planchette

    def _project_pixel(self, u, v, plane_z):
        ray = self._T_wc[:3, :3] @ (np.linalg.inv(self._K) @ np.array([u, v, 1.0]))
        o = self._T_wc[:3, 3]
        if abs(ray[2]) < 1e-9:
            return None
        s = (plane_z - o[2]) / ray[2]
        return o + s * ray if s > 0 else None

    def _world_to_pixel(self, P):
        pc = np.linalg.inv(self._T_wc) @ np.array([*P, 1.0])
        uv = self._K @ pc[:3]
        return uv[0] / uv[2], uv[1] / uv[2]

    def _detect_planchette(self, hsv, board_poly_px):
        """Blue blob inside the board -> (u, v, area, contour, visible_ratio) or None."""
        mask = cv2.inRange(hsv, self._puck_lo, self._puck_hi)
        roi = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(roi, [board_poly_px.astype(np.int32)], 255)
        mask &= roi
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        m = cv2.moments(c)
        if m["m00"] < 20:
            return None
        u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
        # Expected area of the puck seen from the camera (top disc, ignoring the side):
        # pi * (r * f / depth)^2, with the depth of the puck along the optical axis.
        P = self._project_pixel(u, v, self._board_z() + self._puck_height / 2)
        if P is None:
            return None
        depth = (np.linalg.inv(self._T_wc) @ np.array([*P, 1.0]))[2]
        f = 0.5 * (self._K[0, 0] + self._K[1, 1])
        expected = np.pi * (self._puck_radius * f / depth) ** 2
        return u, v, m["m00"], c, float(m["m00"] / expected)

    def _board_z(self):
        return float(self._T_wb[2, 3]) if self._T_wb is not None else BOARD_THICKNESS

    # ------------------------------------------------------------- main loop

    def _cb_image(self, msg: Image):
        if not self._fetch_cam_params():
            self.get_logger().info("waiting for camera_info and TF...", throttle_duration_sec=5.0)
            return
        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        debug = bgr.copy()
        stamp = msg.header.stamp

        est = self._estimate_board(bgr)
        if est is not None:
            T_cb, used, corners, ids, reproj_err = est
            self._T_wb = self._T_wc @ T_cb
            cv2.aruco.drawDetectedMarkers(debug, corners, ids)
            self._publish_board(stamp, reproj_err)
            self._publish_target_markers(stamp)
        if self._T_wb is None:
            self.get_logger().info("board not found yet (need >= 2 ArUco markers)", throttle_duration_sec=5.0)
            self._publish_debug(debug, msg)
            return

        # board outline in the image (used as ROI for the planchette and drawn for debug)
        hx, hy = self._board_size[0] / 2, self._board_size[1] / 2
        outline = np.array([self._world_to_pixel(self._board_to_world(c)) for c in
                            [(hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy)]])
        cv2.polylines(debug, [outline.astype(np.int32)], True, (0, 255, 255), 1)
        for lab, bxy in self._targets.items():
            u, v = self._world_to_pixel(self._board_to_world(bxy))
            cv2.circle(debug, (int(u), int(v)), 2, (0, 0, 255), -1)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        det = self._detect_planchette(hsv, outline)
        pointed = ""
        if det is not None:
            u, v, area, contour, visible = det
            P = self._project_pixel(u, v, self._board_z() + self._puck_height / 2)
            color = (255, 160, 0) if visible >= self._min_visible_ratio else (0, 0, 255)
            cv2.drawContours(debug, [contour], -1, color, 1)
            if P is not None and visible >= self._min_visible_ratio:
                pose = PoseStamped()
                pose.header.stamp, pose.header.frame_id = stamp, self._world_frame
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, P)
                pose.pose.orientation.w = 1.0
                self._pub_puck.publish(pose)
                pointed = self._nearest_target(P)
                cv2.putText(debug, f"planchette ({P[0]:.3f}, {P[1]:.3f}) -> {pointed or '-'}",
                            (int(u) + 12, int(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            else:
                cv2.putText(debug, f"planchette hidden ({visible:.0%} visible)", (int(u) + 12, int(v) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        self._pub_pointed.publish(String(data=pointed))
        self._publish_debug(debug, msg)

    def _nearest_target(self, P):
        best, best_d = "", np.inf
        for lab, bxy in self._targets.items():
            d = np.linalg.norm(self._board_to_world(bxy)[:2] - P[:2])
            if d < best_d:
                best, best_d = lab, d
        return best if best_d <= self._pointed_tolerance else ""

    # ---------------------------------------------------------------- outputs

    def _publish_board(self, stamp, reproj_err):
        msg = PoseStamped()
        msg.header.stamp, msg.header.frame_id = stamp, self._world_frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, self._T_wb[:3, 3])
        qx, qy, qz, qw = _matrix_to_quat(self._T_wb[:3, :3])
        msg.pose.orientation.x, msg.pose.orientation.y = float(qx), float(qy)
        msg.pose.orientation.z, msg.pose.orientation.w = float(qz), float(qw)
        self._pub_board.publish(msg)
        self.get_logger().info(
            f"board at {np.round(self._T_wb[:3, 3], 4)} (reprojection error {reproj_err:.2f} px)",
            throttle_duration_sec=10.0,
        )

    def _publish_target_markers(self, stamp):
        if self._pub_markers.get_subscription_count() == 0:
            return
        arr = MarkerArray()
        for i, (lab, bxy) in enumerate(self._targets.items()):
            m = Marker()
            m.header.stamp, m.header.frame_id = stamp, self._world_frame
            m.ns, m.id, m.type, m.action = "ouija_targets", i, Marker.TEXT_VIEW_FACING, Marker.ADD
            P = self._board_to_world(bxy, 0.002)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, P)
            m.pose.orientation.w = 1.0
            m.scale.z = 0.02
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.1, 0.1, 1.0
            m.text = lab
            arr.markers.append(m)
        arr.markers.append(self._board_outline_marker(stamp, len(arr.markers)))
        self._pub_markers.publish(arr)

    def _board_outline_marker(self, stamp, mid):
        m = Marker()
        m.header.stamp, m.header.frame_id = stamp, self._world_frame
        m.ns, m.id, m.type, m.action = "ouija_targets", mid, Marker.LINE_STRIP, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.003
        m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.8, 0.1, 1.0
        hx, hy = self._board_size[0] / 2, self._board_size[1] / 2
        for c in [(hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy), (hx, hy)]:
            P = self._board_to_world(c, 0.001)
            m.points.append(Point(x=float(P[0]), y=float(P[1]), z=float(P[2])))
        return m

    def _publish_debug(self, debug, msg):
        self._n_frames += 1
        if self._pub_debug.get_subscription_count() > 0 or self._n_frames % 10 == 0:
            out = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            out.header = msg.header
            self._pub_debug.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = OuijaPerceptionNode()
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
