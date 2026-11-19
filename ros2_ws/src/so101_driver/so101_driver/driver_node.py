#!/usr/bin/env python3
"""driver_node.py - ROS2 driver of the SO-ARM101.

The only node that talks to the backend: SO101Sim (MuJoCo) or SO101Follower
(real arm), chosen by the `use_sim` parameter. The rest of the stack never
knows which one runs.

  - Publishes /joint_states in RADIANS (JOINT_STATE_HZ)
  - Subscribes to /joint_command (radians, gripper 0-100 %)
  - Disconnects the backend when the node stops
"""
import math
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import rclpy.node
from geometry_msgs.msg import PoseStamped, TransformStamped
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from so101_sim import SO101Sim
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Gripper joint limits in the URDF (rad).
# LeRobot convention: 0 % = closed (lower), 100 % = open (upper).
GRIPPER_RANGE_RAD = (-0.174533, 1.74533)

# Joint limits of the arm joints, copied from so101_description/urdf (rad).
# Commands outside these ranges are clamped before reaching the backend.
JOINT_LIMITS_RAD = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
}

# Calibration file of the real arm, mounted by docker compose (CALIBRATION_FILE in docker/.env).
CALIBRATION_FILE = Path("/calibration/arm.json")
DEFAULT_PORT = "/dev/ttyACM0"

CONTROL_HZ = 50.0
JOINT_STATE_HZ = 30.0

# External camera (simulation): published on /external_cam/image_raw + camera_info,
# with the camera->world TF (the "provided TF" of the subject) as a static transform.
CAMERA_FRAME = "external_cam_optical"
CAMERA_FRAME_GT = "external_cam_optical_gt"  # sim ground truth, always published (evaluation)
CAMERA_TOPIC = "/external_cam"
# MuJoCo cameras look along -z with y up (OpenGL); OpenCV optical frames look along +z, y down.
MJCF_TO_OPTICAL = np.diag([1.0, -1.0, -1.0, 1.0])

# Ouija scene (scene_ouija.xml) only: where the phase 1 cube is parked, and where the
# planchette starts (on "FIN").
PARKED_OBJECT_POSITION = (-0.30, 0.35, 0.03)
PLANCHETTE_START_POSITION = (0.18, 0.0, 0.0115)  # on "FIN", see ouija/make_board.py


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _gripper_pct_to_rad(pct: float) -> float:
    """0-100 % (LeRobot) -> gripper joint angle in rad (URDF)."""
    lo, hi = GRIPPER_RANGE_RAD
    return lo + _clamp(pct, 0.0, 100.0) / 100.0 * (hi - lo)


class DriverNode(rclpy.node.Node):
    """Driver SO-ARM101: /joint_command -> backend -> /joint_states."""

    def __init__(self):
        super().__init__("so101_driver")
        self._use_sim: bool = self.declare_parameter("use_sim", True).value
        port: str = self.declare_parameter("port", DEFAULT_PORT).value
        mode = "simulation" if self._use_sim else "hardware"
        self.get_logger().info(f"Mode: {mode} (port={port})")

        # Simulation scene (file name in the sim assets dir, empty = default pick & place scene).
        self._scene: str = self.declare_parameter("scene", "").value
        self._cam_w: int = self.declare_parameter("camera_width", 640).value
        self._cam_h: int = self.declare_parameter("camera_height", 480).value
        self._cam_fps: float = self.declare_parameter("camera_fps", 10.0).value
        # Software rendering (osmesa): shadows + reflections cost ~190 ms per 640x480 frame,
        # i.e. 5-7 Hz instead of the 10 Hz announced; ~40 ms without them.
        self._cam_shadows: bool = self.declare_parameter("camera_shadows", False).value
        # Debug/evaluation only: ground-truth object & box poses from the simulator.
        self._publish_gt: bool = self.declare_parameter("publish_ground_truth", True).value
        # Static TF world -> camera from the simulator. False: the TF comes from the camera
        # calibration node (so101_perception camera_calib, ArUco sheet), as on the real robot.
        self._publish_camera_tf: bool = self.declare_parameter("publish_camera_tf", True).value
        # Real arm only. torque=false: the motors stay limp (read /joint_states, move the arm by
        # hand), commands are ignored by the motors. max_step > 0: each command is capped to
        # +/- max_step rad around the present position (arm joints, not the gripper), so a bad
        # command cannot throw the arm (also applies in simulation if set).
        self._torque: bool = self.declare_parameter("torque", True).value
        self._max_step: float = float(self.declare_parameter("max_step", 0.0).value)

        # The backend is not thread-safe: the camera thread renders while the
        # timers step the physics, so every backend access takes this lock.
        self._backend_lock = threading.Lock()

        self._robot = self._connect_arm(port)
        self.get_logger().info(f"Robot connected: {type(self._robot).__name__}")

        self._last_obs: dict[str, float] = {}  # last positions read ("<joint>.pos"), for max_step
        # Last command received, "<joint>.pos" -> value (rad, gripper in %).
        # Only the joints named in the last message are present: a partial
        # command moves only those joints.
        self._pending_action: dict[str, float] = {}

        self._pub_joint_states = self.create_publisher(JointState, "/joint_states", 10)
        self._sub_joint_command = self.create_subscription(
            JointState, "/joint_command", self._cb_joint_command, 10
        )
        self._control_timer = self.create_timer(1.0 / CONTROL_HZ, self._control_step)
        self._state_timer = self.create_timer(1.0 / JOINT_STATE_HZ, self._publish_joint_states)

        self._camera_thread = None
        self._camera_stop = threading.Event()
        if self._use_sim:
            self._start_sim_camera()
            # Simulation only: put the cube back at a random reachable pose.
            self.create_service(Trigger, "/sim/randomize_object", self._srv_randomize_object)
            if self._has_planchette:
                # Ouija scene: SO101Sim.connect() randomised the cube onto the board, park it.
                self._park_object()
                self.create_service(Trigger, "/sim/reset_planchette", self._srv_reset_planchette)

        self.get_logger().info("Driver node ready.")

    def _srv_randomize_object(self, req, res):
        with self._backend_lock:
            self._robot.randomize_object()
            pos = self._robot.get_object_position()
        res.success = True
        res.message = f"object moved to ({pos[0]:.3f}, {pos[1]:.3f})"
        self.get_logger().info(res.message)
        return res

    def _set_free_body(self, joint_name: str, position, settle_steps: int = 100):
        """Teleports a free body (sim only) and lets the physics settle. Caller holds the lock."""
        import mujoco

        adr = self._robot.model.joint(joint_name).qposadr[0]
        self._robot.data.qpos[adr:adr + 3] = position
        self._robot.data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self._robot.data.qvel[:] = 0
        for _ in range(settle_steps):
            mujoco.mj_step(self._robot.model, self._robot.data)

    def _park_object(self):
        with self._backend_lock:
            self._set_free_body("pickup_object_free", PARKED_OBJECT_POSITION)
        self.get_logger().info("Ouija scene: pick & place cube parked out of the board.")

    def _srv_reset_planchette(self, req, res):
        with self._backend_lock:
            self._set_free_body("planchette_free", PLANCHETTE_START_POSITION)
        res.success = True
        res.message = "planchette back to its start position"
        return res

    # ------------------------------------------------------------- camera (sim)

    def _start_sim_camera(self):
        """Publishes the MuJoCo external camera from a dedicated thread (rendering is slow)."""
        # Camera frames are sensor data: QoS "sensor data" = Best Effort, keep last 5. A lost
        # frame is replaced by the next one 100 ms later, a retransmitted one arrives too late.
        # Subscribers must be Best Effort too (a Reliable subscriber gets nothing from a
        # Best Effort publisher). camera_info is small and must arrive: it stays Reliable.
        self._pub_image = self.create_publisher(Image, f"{CAMERA_TOPIC}/image_raw", qos_profile_sensor_data)
        self._pub_cam_info = self.create_publisher(CameraInfo, f"{CAMERA_TOPIC}/camera_info", 5)
        import mujoco

        self._has_planchette = mujoco.mj_name2id(self._robot.model, mujoco.mjtObj.mjOBJ_BODY, "planchette") >= 0
        if self._publish_gt:
            self._pub_gt_object = self.create_publisher(PoseStamped, "/debug/object_position_gt", 5)
            self._pub_gt_box = self.create_publisher(PoseStamped, "/debug/drop_box_position_gt", 5)
            self._pub_gt_puck = self.create_publisher(PoseStamped, "/debug/planchette_position_gt", 5)
            # the same ground truth as RViz markers (red cube mesh, box, planchette): display only
            self._pub_gt_markers = self.create_publisher(MarkerArray, "/debug/sim_markers", 5)

        # Static TF world -> camera optical frame, from the simulator (ground truth). The
        # ground truth is always published as CAMERA_FRAME_GT (to evaluate the calibration);
        # the frame used by perception (CAMERA_FRAME) only if publish_camera_tf.
        T = self._robot.get_camera_extrinsics() @ MJCF_TO_OPTICAL
        self._tf_static = StaticTransformBroadcaster(self)
        tfs = [self._matrix_to_tf(T, "world", CAMERA_FRAME_GT)]
        if self._publish_camera_tf:
            tfs.append(self._matrix_to_tf(T, "world", CAMERA_FRAME))
        else:
            self.get_logger().info(f"TF world -> {CAMERA_FRAME} not published (publish_camera_tf=false): "
                                   "expected from the camera_calib node")
        self._tf_static.sendTransform(tfs)

        K = self._robot.get_camera_intrinsics()
        self._cam_info = CameraInfo()
        self._cam_info.header.frame_id = CAMERA_FRAME
        self._cam_info.width, self._cam_info.height = self._cam_w, self._cam_h
        self._cam_info.distortion_model = "plumb_bob"
        self._cam_info.d = [0.0] * 5
        self._cam_info.k = [float(v) for v in K.flatten()]
        self._cam_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self._cam_info.p = [float(v) for v in np.hstack([K, np.zeros((3, 1))]).flatten()]

        self._camera_thread = threading.Thread(target=self._camera_loop, name="sim_camera", daemon=True)
        self._camera_thread.start()
        self.get_logger().info(
            f"Sim camera: {self._cam_w}x{self._cam_h} @ {self._cam_fps:.0f} Hz on {CAMERA_TOPIC}/image_raw"
        )

    def _camera_loop(self):
        import mujoco  # only needed in simulation

        # The OpenGL context belongs to this thread: create the renderer here.
        renderer = mujoco.Renderer(self._robot.model, height=self._cam_h, width=self._cam_w)
        period = 1.0 / max(self._cam_fps, 0.1)
        while not self._camera_stop.is_set() and rclpy.ok():
            t0 = time.monotonic()
            try:
                with self._backend_lock:  # update_scene reads the physics state (fast)
                    renderer.update_scene(self._robot.data, camera=self._robot.camera_name)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self._cam_shadows
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self._cam_shadows
                    if self._publish_gt:
                        gt_obj = self._robot.get_object_position()
                        gt_box = self._robot.get_drop_box_position()
                        gt_puck = self._robot.data.body("planchette").xpos.copy() if self._has_planchette else None
                        gt_markers = self._ground_truth_markers()
                rgb = renderer.render()  # slow part, outside the lock
                stamp = self.get_clock().now().to_msg()
                msg = Image()
                msg.header.stamp = stamp
                msg.header.frame_id = CAMERA_FRAME
                msg.height, msg.width = rgb.shape[0], rgb.shape[1]
                msg.encoding = "rgb8"
                msg.is_bigendian = False
                msg.step = rgb.shape[1] * 3
                msg.data = rgb.tobytes()
                self._cam_info.header.stamp = stamp
                self._pub_image.publish(msg)
                self._pub_cam_info.publish(self._cam_info)
                if self._publish_gt:
                    self._pub_gt_object.publish(self._pose_msg(gt_obj, stamp))
                    self._pub_gt_box.publish(self._pose_msg(gt_box, stamp))
                    if gt_puck is not None:
                        self._pub_gt_puck.publish(self._pose_msg(gt_puck, stamp))
                    for m in gt_markers.markers:
                        m.header.stamp = stamp
                    self._pub_gt_markers.publish(gt_markers)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"camera: {exc}", throttle_duration_sec=5.0)
            self._camera_stop.wait(max(0.0, period - (time.monotonic() - t0)))
        renderer.close()

    # Ground truth of the scene as RViz markers (simulation only). RViz knows nothing about the
    # MuJoCo scene: without these the arm grasps "nothing" on screen. Display/evaluation only, the
    # pipeline keeps using the perception (/object_position).
    GT_MESHES = {  # body -> (STL in the image, scale, offset of the mesh origin in the body frame)
        "pickup_object": ("file:///opt/so101/sim/so101_sim/assets/so101/assets/hollow_cube_rhombus_50mm.stl",
                          0.001, (0.0, 0.0, -0.025)),
    }
    GT_BODIES = ("pickup_object", "drop_box", "planchette")

    def _ground_truth_markers(self) -> MarkerArray:
        import mujoco

        model, data = self._robot.model, self._robot.data
        out = MarkerArray()
        for body in self.GT_BODIES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if bid < 0:
                continue
            R = data.xmat[bid].reshape(3, 3)
            if body in self.GT_MESHES:
                uri, scale, off = self.GT_MESHES[body]
                geoms = self._body_geoms(model, bid)
                rgba = self._geom_rgba(model, geoms[0]) if geoms else (0.9, 0.1, 0.1, 1.0)
                m = self._marker(body, 0, Marker.MESH_RESOURCE, data.xpos[bid] + R @ off, R, (scale,) * 3, rgba)
                m.mesh_resource = uri
                out.markers.append(m)
                continue
            for k, gid in enumerate(self._body_geoms(model, bid)):
                gtype = model.geom_type[gid]
                size = model.geom_size[gid]
                if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                    shape, scale = Marker.CUBE, 2.0 * size
                elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    shape, scale = Marker.CYLINDER, (2.0 * size[0], 2.0 * size[0], 2.0 * size[1])
                elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                    shape, scale = Marker.SPHERE, (2.0 * size[0],) * 3
                else:
                    continue  # meshes of other bodies: not displayed
                out.markers.append(self._marker(body, k, shape, data.geom_xpos[gid], data.geom_xmat[gid].reshape(3, 3),
                                                scale, self._geom_rgba(model, gid)))
        return out

    @staticmethod
    def _body_geoms(model, bid):
        return [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid and model.geom_group[g] < 3]

    @staticmethod
    def _geom_rgba(model, gid):
        mat = model.geom_matid[gid]
        return model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[gid]

    @staticmethod
    def _marker(ns, mid, shape, pos, R, scale, rgba) -> Marker:
        m = Marker()
        m.header.frame_id = "world"
        m.ns, m.id, m.type, m.action = f"gt/{ns}", mid, shape, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, pos)
        # rotation matrix -> quaternion (w, x, y, z), MuJoCo style
        w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
        if w > 1e-6:
            x, y, z = (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)
        else:  # 180 degree rotation: fall back to the dominant diagonal term
            i = int(np.argmax(np.diag(R)))
            q = np.zeros(4)
            q[i] = math.sqrt(max(0.0, 1.0 + 2 * R[i, i] - R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
            j, k = (i + 1) % 3, (i + 2) % 3
            q[j], q[k] = (R[j, i] + R[i, j]) / (4 * q[i]), (R[k, i] + R[i, k]) / (4 * q[i])
            x, y, z = q
        m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z = (
            float(w), float(x), float(y), float(z))
        m.scale.x, m.scale.y, m.scale.z = map(float, scale)
        m.color.r, m.color.g, m.color.b, m.color.a = map(float, rgba)
        m.lifetime.sec = 1
        return m

    def _pose_msg(self, xyz, stamp) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, xyz)
        msg.pose.orientation.w = 1.0
        return msg

    @staticmethod
    def _matrix_to_tf(T, parent, child) -> TransformStamped:
        R = T[:3, :3]
        # rotation matrix -> quaternion (w, x, y, z), robust branch selection
        tr = np.trace(R)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
        tf = TransformStamped()
        tf.header.frame_id = parent
        tf.child_frame_id = child
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, T[:3, 3])
        tf.transform.rotation.x, tf.transform.rotation.y = float(x), float(y)
        tf.transform.rotation.z, tf.transform.rotation.w = float(z), float(w)
        return tf

    def _connect_arm(self, port: str = DEFAULT_PORT) -> SO101Sim | SO101Follower:
        """Creates and connects the backend chosen by use_sim (provided)."""
        if self._use_sim:
            kwargs = {"camera_width": self._cam_w, "camera_height": self._cam_h}
            if self._scene:
                from so101_sim import ASSETS_DIR

                kwargs["scene_path"] = ASSETS_DIR / self._scene
                self.get_logger().info(f"Scene: {kwargs['scene_path']}")
            robot = SO101Sim(**kwargs)
            robot.connect()
            return robot

        if not CALIBRATION_FILE.is_file():
            raise RuntimeError(
                f"No calibration file at {CALIBRATION_FILE}: set CALIBRATION_FILE in docker/.env"
            )
        config = SO101FollowerConfig(
            port=port,
            id=CALIBRATION_FILE.stem,
            calibration_dir=CALIBRATION_FILE.parent,
            use_radians=True,
        )
        robot = SO101Follower(config)
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            robot.disconnect()
            raise RuntimeError(
                f"{CALIBRATION_FILE} does not match the motors: wrong file for this arm? "
                "Re-run lerobot-calibrate on the host."
            )
        if self._torque:
            self.get_logger().warn(
                "Torque ENABLED: the arm holds its position and follows /joint_command. It goes limp "
                "(and may fall) when this node stops: bring it to its rest pose first."
            )
        else:
            robot.bus.disable_torque()
            self.get_logger().warn("Torque disabled (torque=false): the arm can be moved by hand, "
                                   "/joint_command has no effect.")
        if self._max_step > 0:
            self.get_logger().info(f"Commands capped to +/- {self._max_step:.3f} rad per step (max_step)")
        return robot

    def destroy_node(self):
        """Disconnects the backend (provided). On the real arm this disables the torque."""
        self._camera_stop.set()
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=2.0)
        if self._robot.is_connected:
            self._robot.disconnect()
            self.get_logger().info("Robot disconnected.")
        super().destroy_node()

    def _cb_joint_command(self, msg: JointState):
        if len(msg.name) != len(msg.position):
            self.get_logger().warn(
                f"/joint_command ignored: {len(msg.name)} names for {len(msg.position)} positions"
            )
            return

        action: dict[str, float] = {}
        for name, value in zip(msg.name, msg.position):
            if name == "gripper":
                target = _clamp(float(value), 0.0, 100.0)
            elif name in JOINT_LIMITS_RAD:
                target = _clamp(float(value), *JOINT_LIMITS_RAD[name])
            else:
                self.get_logger().warn(
                    f"/joint_command: unknown joint '{name}' ignored", throttle_duration_sec=1.0
                )
                continue
            if target != value:
                self.get_logger().warn(
                    f"/joint_command: {name}={value:.3f} clamped to {target:.3f}",
                    throttle_duration_sec=1.0,
                )
            action[f"{name}.pos"] = target

        if action:
            # Merge with a command not yet sent, so that two partial commands
            # arriving between two control steps are both applied.
            self._pending_action.update(action)

    def _control_step(self):
        if not self._pending_action or self._robot is None:
            return
        action, self._pending_action = self._pending_action, {}
        if self._max_step > 0 and self._last_obs:
            action = self._cap_action(action)
        try:
            with self._backend_lock:
                self._robot.send_action(action)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"send_action failed: {exc}", throttle_duration_sec=1.0)

    def _cap_action(self, action: dict[str, float]) -> dict[str, float]:
        """Clamps each target to +/- max_step around the last position read (safety)."""
        capped = {}
        for key, target in action.items():
            present = self._last_obs.get(key)
            if present is None or key == "gripper.pos":  # gripper: torque-limited by LeRobot, sent in one go
                capped[key] = target
                continue
            cap = self._max_step
            capped[key] = _clamp(target, present - cap, present + cap)
            if abs(capped[key] - target) > 1e-6:
                self.get_logger().warn(
                    f"{key}: target {target:.3f} capped to {capped[key]:.3f} (max_step)", throttle_duration_sec=1.0
                )
        return capped

    def _publish_joint_states(self):
        if self._robot is None:
            return
        try:
            with self._backend_lock:
                obs = self._robot.get_observation()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"get_observation failed: {exc}", throttle_duration_sec=1.0)
            return

        self._last_obs = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for name in JOINT_NAMES:
            key = f"{name}.pos"
            if key not in obs:
                continue
            value = float(obs[key])
            if name == "gripper":
                value = _gripper_pct_to_rad(value)  # the URDF joint is in rad
            msg.name.append(name)
            msg.position.append(value)
        self._pub_joint_states.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DriverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
