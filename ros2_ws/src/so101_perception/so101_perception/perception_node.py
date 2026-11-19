#!/usr/bin/env python3
"""perception_node.py - detects the red cube and the drop box, publishes their 3D position.

Pipeline (see J3 slides "from pixel to 3D"):
  image -> HSV segmentation -> blob centroid (u, v)
        -> ray = K^-1 [u v 1] (K from /external_cam/camera_info)
        -> ray expressed in world with the camera TF (tf2: world -> camera optical frame)
        -> intersection with a known horizontal plane (z = object height / 2)
Publishes /object_position and /drop_box_position (geometry_msgs/PoseStamped, frame
world) at the image rate, plus /perception/debug_image for RViz.
"""
import os
import time
from pathlib import Path

import numpy as np
import cv2
import rclpy
import rclpy.node
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Point
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker


WORKSPACE_FILE = Path("/ros2_ws/calibration/workspace.yaml")


def load_workspace():
    """Real-robot workspace settings (calibration/workspace.yaml), {} in simulation or if absent."""
    if os.environ.get("USE_SIM", "true").lower() == "true" or not WORKSPACE_FILE.is_file():
        return {}
    import yaml

    return yaml.safe_load(WORKSPACE_FILE.read_text()) or {}


class PerceptionNode(rclpy.node.Node):
    """Red cube + drop box detection, pixel -> 3D by ray/plane intersection."""

    def __init__(self):
        super().__init__("so101_perception")
        p = self.declare_parameter
        self._camera_topic = p("camera_topic", "/external_cam").value
        self._world_frame = p("world_frame", "world").value
        # Height of the plane used to intersect the viewing ray, per target (m).
        # Plane z(x) = table_z0 + table_slope * (x - table_x0) + h (the table can be tilted in the robot
        # frame on the real arm, see calibration/workspace.yaml); h = plane_z below. Sim: z = h.
        ws = load_workspace()
        tb = ws.get("table", {})
        self._table_x0 = float(tb.get("x0", 0.0))
        self._table_z0 = float(tb.get("z0", 0.0))
        self._table_slope = float(tb.get("slope", 0.0))
        # Detections closer than this to the robot base (m) are ignored: on the real arm the red clamp of
        # the base looks like the cube. 0 = off (simulation).
        self._ignore_radius = float(p("ignore_radius", float(ws.get("ignore_radius", 0.0))).value)
        # Offset (x, y) added to the camera positions to express them in the arm's frame (real arm).
        self._xy_offset = np.array([*[float(v) for v in ws.get("xy_offset", [0.0, 0.0])], 0.0])
        self._object_plane_z = p("object_plane_z", float(ws.get("object_plane_h", 0.025))).value  # cube centre
        self._box_plane_z = p("box_plane_z", float(ws.get("box_plane_h", 0.025))).value
        self._min_area = p("min_blob_area_px", 40).value
        # HSV thresholds (OpenCV: H 0-179). Red wraps around 0, hence two ranges.
        self._obj_lo1 = np.array(p("object_hsv_low1", [160, 90, 60]).value, dtype=np.uint8)
        self._obj_hi1 = np.array(p("object_hsv_high1", [179, 255, 255]).value, dtype=np.uint8)
        self._obj_lo2 = np.array(p("object_hsv_low2", [0, 90, 60]).value, dtype=np.uint8)
        self._obj_hi2 = np.array(p("object_hsv_high2", [8, 255, 255]).value, dtype=np.uint8)
        self._box_lo = np.array(p("box_hsv_low", ws.get("box_hsv_low", [5, 80, 40])).value, dtype=np.uint8)
        self._box_hi = np.array(p("box_hsv_high", ws.get("box_hsv_high", [22, 255, 160])).value, dtype=np.uint8)

        self._bridge = CvBridge()
        self._K = None
        self._camera_frame = None
        self._T_wc = None  # 4x4 world <- camera optical
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, f"{self._camera_topic}/camera_info", self._cb_camera_info, 5)
        # Best Effort, like the camera publisher (sim driver or real camera): see README "QoS".
        self.create_subscription(Image, f"{self._camera_topic}/image_raw", self._cb_image, qos_profile_sensor_data)
        self._pub_object = self.create_publisher(PoseStamped, "/object_position", 5)
        self._pub_box = self.create_publisher(PoseStamped, "/drop_box_position", 5)
        self._pub_debug = self.create_publisher(Image, "/perception/debug_image", qos)
        self._pub_marker = self.create_publisher(Marker, "/object_marker", 5)
        self._n_frames = 0
        self.get_logger().info("Perception node ready.")

    # ----------------------------------------------------------------- inputs

    def _cb_camera_info(self, msg: CameraInfo):
        if msg.k[0] <= 0.0:  # real camera without intrinsics file: K is all zeros
            self.get_logger().warning("camera_info without intrinsics (fx = 0): calibrate the camera "
                                      "(ros2_ws/calibration/camera_info.yaml)", throttle_duration_sec=10.0)
            return
        self._K = np.array(msg.k, dtype=float).reshape(3, 3)
        self._camera_frame = msg.header.frame_id

    def _fetch_cam_params(self) -> bool:
        """K from CameraInfo, camera pose from tf2 (static TF published by the driver)."""
        if self._K is None or not self._camera_frame:
            return False
        # Re-read the camera pose every 2 s: it changes when the camera is recalibrated or when
        # the simulator restarts with another scene (the Ouija scene has its own camera).
        now = time.monotonic()
        if self._T_wc is None or now - getattr(self, "_T_wc_time", 0.0) > 2.0:
            try:
                tf = self._tf_buffer.lookup_transform(self._world_frame, self._camera_frame, rclpy.time.Time())
            except Exception:  # noqa: BLE001 - TF not available yet
                return self._T_wc is not None
            t, q = tf.transform.translation, tf.transform.rotation
            T = np.eye(4)
            T[:3, :3] = self._quat_to_matrix(q.x, q.y, q.z, q.w)
            T[:3, 3] = [t.x, t.y, t.z]
            if self._T_wc is None or np.max(np.abs(T - self._T_wc)) > 1e-6:
                what = "received" if self._T_wc is None else "updated"
                self._T_wc = T
                self.get_logger().info(f"camera pose {what}: t={np.round(T[:3, 3], 3)}")
            self._T_wc_time = now
        return True

    @staticmethod
    def _quat_to_matrix(x, y, z, w):
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    # ------------------------------------------------------------- processing

    def _cb_image(self, msg: Image):
        if not self._fetch_cam_params():
            self.get_logger().info("waiting for camera_info and TF...", throttle_duration_sec=5.0)
            return
        rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        obj_mask = cv2.inRange(hsv, self._obj_lo1, self._obj_hi1) | cv2.inRange(hsv, self._obj_lo2, self._obj_hi2)
        box_mask = cv2.inRange(hsv, self._box_lo, self._box_hi)
        obj = self._detect_object(obj_mask)
        box = self._detect_object(box_mask)

        debug = rgb.copy()
        if obj is not None and self._ignore_radius > 0.0:  # real arm: the red clamp of the base
            P0 = self._project_to_3d(*obj[:2], self._object_plane_z)
            if P0 is not None and float(np.hypot(P0[0], P0[1])) < self._ignore_radius:
                obj = None
        if obj is not None and self._touches_border(obj[3], rgb.shape):
            # a blob cut by the image border has a wrong centroid: do not publish it
            self.get_logger().warn("object partly out of the camera view, detection skipped", throttle_duration_sec=2.0)
            self._annotate(debug, obj, None, (255, 80, 80), "object (bord)")
            obj = None
        if obj is not None:
            P = self._project_to_3d(*obj[:2], self._object_plane_z)
            P = None if P is None else P + self._xy_offset
            self._publish_result(self._pub_object, P, msg.header.stamp)
            self._publish_marker(P, msg.header.stamp)
            self._annotate(debug, obj, P, (0, 255, 0), "object")
        if box is not None:
            P = self._project_to_3d(*box[:2], self._box_plane_z)
            P = None if P is None else P + self._xy_offset
            self._publish_result(self._pub_box, P, msg.header.stamp)
            self._annotate(debug, box, P, (0, 200, 255), "box")

        self._n_frames += 1
        if self._pub_debug.get_subscription_count() > 0 or self._n_frames % 10 == 0:
            out = self._bridge.cv2_to_imgmsg(debug, encoding="rgb8")
            out.header = msg.header
            self._pub_debug.publish(out)

    @staticmethod
    def _touches_border(bbox, shape, margin=2):
        x, y, w, h = bbox
        return x <= margin or y <= margin or x + w >= shape[1] - margin or y + h >= shape[0] - margin

    def _detect_object(self, mask):
        """Largest blob of the mask -> (u, v, area, bbox) or None."""
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        m = cv2.moments(c)
        if m["m00"] < self._min_area:
            return None
        return m["m10"] / m["m00"], m["m01"] / m["m00"], m["m00"], cv2.boundingRect(c)

    def _project_to_3d(self, u, v, plane_h):
        """Pixel -> world point on the plane z = table(x) + plane_h (horizontal in simulation)."""
        ray_cam = np.linalg.inv(self._K) @ np.array([u, v, 1.0])
        d = self._T_wc[:3, :3] @ ray_cam
        o = self._T_wc[:3, 3]
        den = d[2] - self._table_slope * d[0]
        if abs(den) < 1e-9:
            return None
        t = (self._table_z0 + plane_h + self._table_slope * (o[0] - self._table_x0) - o[2]) / den
        return o + t * d if t > 0 else None

    def _publish_result(self, pub, P, stamp):
        if P is None:
            return
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self._world_frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, P)
        msg.pose.orientation.w = 1.0
        pub.publish(msg)

    def _publish_marker(self, P, stamp):
        """Green 5 cm wireframe cube in RViz at the detected object position (wireframe: the
        simulation's ground-truth cube, /debug/sim_markers, stays visible inside it)."""
        if P is None:
            return
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self._world_frame
        m.ns, m.id, m.type, m.action = "object", 0, Marker.LINE_LIST, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, P)
        m.pose.orientation.w = 1.0
        m.scale.x = 0.003
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.2, 0.9
        h = 0.025
        corners = [(sx * h, sy * h, sz * h) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        for i, a in enumerate(corners):
            for j in range(i + 1, 8):
                b = corners[j]
                if sum(x != y for x, y in zip(a, b)) == 1:  # edges: corners differing in one axis
                    m.points.append(Point(x=a[0], y=a[1], z=a[2]))
                    m.points.append(Point(x=b[0], y=b[1], z=b[2]))
        m.lifetime.sec = 1
        self._pub_marker.publish(m)

    @staticmethod
    def _annotate(img, det, P, color, label):
        u, v, _, (x, y, w, h) = det
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        cv2.circle(img, (int(u), int(v)), 3, color, -1)
        txt = label if P is None else f"{label} ({P[0]:.3f}, {P[1]:.3f})"
        cv2.putText(img, txt, (x, max(10, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
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
