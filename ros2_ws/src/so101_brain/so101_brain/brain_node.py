#!/usr/bin/env python3
"""brain_node.py - intelligence of the arm: kinematics, then pick & place.

Never talks to the backend: reads /joint_states, commands through /joint_command.

  - Forward kinematics (pinocchio, from the URDF): publishes the TCP pose on
    /end_effector_pose (geometry_msgs/PoseStamped, frame "world").
  - Inverse kinematics: service /go_to_target (so101_interfaces/GoToTarget).
    Position in the world frame; the gripper approaches from above by default
    (TCP z axis pointing down), or along the z axis of the requested orientation.
  - Service /set_joint_positions (so101_interfaces/SetJointPositions):
    5 arm joints in DEGREES, optional 6th value = gripper 0-100 %.
  - Service /move_line (so101_interfaces/MoveLine): straight-line Cartesian motion of the
    TCP (IK on 1 cm waypoints, seeded with the previous one), used to push the planchette.
  - Every motion is interpolated at CONTROL_HZ with a max joint speed.
  - Pick & place: service /pick_and_place (std_srvs/Trigger), explicit state machine
    OBSERVE -> APPROACH -> DESCEND -> GRASP -> LIFT -> TRANSPORT -> RELEASE -> RETREAT,
    using the last known /object_position and /drop_box_position (never blocks on
    perception), re-plans if the object moves, retries a failed grasp. /brain/stop aborts.
    Current state on /brain/state (std_msgs/String).
"""
import math
import threading
import time

import numpy as np
import pinocchio as pin
import rclpy
import rclpy.node
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from sensor_msgs.msg import JointState
from so101_interfaces.srv import GoToTarget, MoveLine, SetJointPositions
from std_msgs.msg import String
from std_srvs.srv import Trigger

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
JOINT_NAMES = ARM_JOINT_NAMES + ["gripper"]

TCP_FRAME = "gripper_frame_link"  # tool center point, its z axis points out of the gripper
URDF_FALLBACK = "/ros2_ws/src/so101_description/urdf/so101.urdf"

CONTROL_HZ = 50.0
POSE_HZ = 20.0
MAX_JOINT_SPEED = 1.0  # rad/s, applied to the fastest joint of a motion (default of `max_joint_speed`)
MIN_MOTION_DURATION = 0.3  # s
LINE_STEP = 0.01  # m, waypoint spacing of /move_line
LINE_DEFAULT_SPEED = 0.05  # m/s
POSITION_TOLERANCE = 0.01  # m, success criterion of /go_to_target (subject: < 1 cm)
IK_POSITION_EPS = 1e-3  # m, convergence of the solver
IK_AXIS_EPS = 0.15  # |z_des - z_tcp| ~ 8.6 degrees
IK_LIMIT_MARGIN = 0.035  # rad, keep IK solutions away from the joint limits (servos sag there)


def _urdf_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        return get_package_share_directory("so101_description") + "/urdf/so101.urdf"
    except Exception:  # noqa: BLE001
        return URDF_FALLBACK


class BrainNode(rclpy.node.Node):
    """Brain of the arm: kinematics, then pick & place."""

    def __init__(self):
        super().__init__("so101_brain")

        # ---- kinematic model
        self._model = pin.buildModelFromUrdf(_urdf_path())
        self._data = self._model.createData()
        self._tcp_id = self._model.getFrameId(TCP_FRAME)
        self._q_lo = np.array(self._model.lowerPositionLimit[:5]) + IK_LIMIT_MARGIN
        self._q_hi = np.array(self._model.upperPositionLimit[:5]) - IK_LIMIT_MARGIN

        # ---- state (updated by /joint_states)
        self._lock = threading.Lock()
        self._q = np.zeros(6)  # 5 arm joints (rad) + gripper (rad, URDF convention)
        self._have_state = False

        # ---- motion in progress: piecewise-linear joint trajectory
        # _traj_points: list of (time offset s, q_arm); None when idle.
        self._traj_points = None
        self._traj_t0 = 0.0

        cb_io = MutuallyExclusiveCallbackGroup()
        cb_srv = ReentrantCallbackGroup()

        self._sub_js = self.create_subscription(
            JointState, "/joint_states", self._cb_joint_states, 10, callback_group=cb_io
        )
        self._pub_cmd = self.create_publisher(JointState, "/joint_command", 10)
        self._pub_pose = self.create_publisher(PoseStamped, "/end_effector_pose", 10)
        self._pub_target = self.create_publisher(PoseStamped, "/ik_target", 10)

        self.create_timer(1.0 / CONTROL_HZ, self._control_step, callback_group=cb_io)
        self.create_timer(1.0 / POSE_HZ, self._publish_end_effector_pose, callback_group=cb_io)

        self.create_service(GoToTarget, "/go_to_target", self._srv_go_to_target, callback_group=cb_srv)
        self.create_service(
            SetJointPositions, "/set_joint_positions", self._srv_set_joint_positions, callback_group=cb_srv
        )
        self.create_service(MoveLine, "/move_line", self._srv_move_line, callback_group=cb_srv)

        # ---- pick & place
        pp = self.declare_parameter
        # Real arm: lower it for the first tests (ARM_MAX_SPEED in docker/.env).
        self._max_joint_speed = float(pp("max_joint_speed", MAX_JOINT_SPEED).value)
        self._observe_joints_deg = pp("observe_joints_deg", [0.0, -80.0, 80.0, 60.0, 0.0]).value
        self._approach_height = pp("approach_height", 0.12).value
        self._grasp_height = pp("grasp_height", 0.03).value
        self._lift_height = pp("lift_height", 0.15).value
        self._drop_height = pp("drop_height", 0.10).value
        self._max_target_age = pp("max_target_age", 2.0).value  # s, freshness of a detection
        self._max_retries = pp("max_grasp_retries", 2).value
        self._replan_distance = pp("replan_distance", 0.02).value  # m, object moved -> re-approach
        self._motion_tolerance = pp("motion_tolerance", 0.03).value  # m, accepted error of a step

        self._targets = {}  # "object" / "box" -> (xyz, monotonic time)
        self._abort = threading.Event()
        self._task_running = threading.Lock()
        self.create_subscription(
            PoseStamped, "/object_position", lambda m: self._cb_target("object", m), 5, callback_group=cb_io
        )
        self.create_subscription(
            PoseStamped, "/drop_box_position", lambda m: self._cb_target("box", m), 5, callback_group=cb_io
        )
        self._pub_state = self.create_publisher(String, "/brain/state", 5)
        self.create_service(Trigger, "/pick_and_place", self._srv_pick_and_place, callback_group=cb_srv)
        self.create_service(Trigger, "/brain/stop", self._srv_stop, callback_group=cb_srv)
        self._set_state("IDLE")

        self.get_logger().info("Brain node ready.")

    # ------------------------------------------------------------------ state

    def _cb_joint_states(self, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name in JOINT_NAMES:
                    self._q[JOINT_NAMES.index(name)] = pos
            self._have_state = True

    def _current_q(self) -> np.ndarray:
        with self._lock:
            return self._q.copy()

    # --------------------------------------------------------------- kinematics

    def _forward_kinematics(self, q) -> pin.SE3:
        """TCP placement in the world frame for q (6 values, rad)."""
        q = np.asarray(q, dtype=float)
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        return self._data.oMf[self._tcp_id].copy()

    def _publish_end_effector_pose(self):
        if not self._have_state:
            return
        T = self._forward_kinematics(self._current_q())
        self._pub_pose.publish(self._se3_to_msg(T))

    def _se3_to_msg(self, T: pin.SE3) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, T.translation)
        qx, qy, qz, qw = pin.Quaternion(T.rotation).coeffs()
        msg.pose.orientation.x, msg.pose.orientation.y = float(qx), float(qy)
        msg.pose.orientation.z, msg.pose.orientation.w = float(qz), float(qw)
        return msg

    def _solve_ik(self, target_position, approach_dir=(0.0, 0.0, -1.0), q_seed=None, x_dir=None):
        """Damped least squares IK on the 5 arm joints.

        Tasks: TCP position (3) + TCP z axis aligned with approach_dir (2 effective).
        x_dir (optional): desired direction of the TCP x axis (the jaw axis), obtained
        afterwards with the wrist roll, which is free when the gripper points down.
        Returns (q_arm, pos_error_m, axis_error) for the best solution found.
        """
        q_arm, pos_err, axis_err = self._solve_ik_position_axis(target_position, approach_dir, q_seed)
        if q_arm is not None and x_dir is not None:
            q_arm = self._align_roll(q_arm, x_dir)
        return q_arm, pos_err, axis_err

    def _align_roll(self, q_arm, x_dir):
        """Turns the wrist roll so that the TCP x axis (jaw axis) points along x_dir
        (projected on the plane perpendicular to the TCP z axis). The jaws are symmetric,
        so an angle differing by 180 degrees is equivalent: the smaller rotation is used."""
        x_des = np.asarray(x_dir, dtype=float)
        q = np.concatenate([q_arm, [0.0]])

        def error(q):
            R = self._forward_kinematics(q).rotation
            x, z = R[:, 0], R[:, 2]
            xd = x_des - z * np.dot(x_des, z)
            if np.linalg.norm(xd) < 1e-6:
                return 0.0
            xd /= np.linalg.norm(xd)
            e = math.atan2(np.dot(np.cross(x, xd), z), np.dot(x, xd))
            return (e + math.pi / 2) % math.pi - math.pi / 2  # symmetric jaws

        e0 = error(q)
        q[4] = np.clip(q[4] + e0, self._q_lo[4], self._q_hi[4])
        if abs(error(q)) > abs(e0) * 0.5 + 1e-3:  # roll axis had the other sign
            q[4] = np.clip(q_arm[4] - e0, self._q_lo[4], self._q_hi[4])
        return q[:5].copy()

    def _solve_ik_position_axis(self, target_position, approach_dir, q_seed):
        p_des = np.asarray(target_position, dtype=float)
        z_des = np.asarray(approach_dir, dtype=float)
        z_des = z_des / np.linalg.norm(z_des)
        q_now = self._current_q() if q_seed is None else np.asarray(q_seed, dtype=float)

        pan = math.atan2(p_des[1], p_des[0])
        seeds = [
            q_now[:5],
            np.array([pan, -0.3, 0.8, 0.6, 0.0]),
            np.array([pan, 0.4, 1.2, 0.0, 0.0]),
            np.array([pan, -1.0, 1.5, 1.0, 0.0]),
        ]
        rng = np.random.default_rng(0)
        for _ in range(6):
            s = rng.uniform(self._q_lo, self._q_hi)
            s[0] = pan
            seeds.append(s)

        best = (None, np.inf, np.inf)
        q = q_now.copy()
        for axis_weight in (0.05, 0.01):  # strict approach first, then mostly position
            for seed in seeds:
                q[:5] = np.clip(seed, self._q_lo, self._q_hi)
                for _ in range(150):
                    T = self._forward_kinematics(q)
                    z = T.rotation[:, 2]
                    e_p = p_des - T.translation
                    e_z = z_des - z
                    if np.linalg.norm(e_p) < IK_POSITION_EPS and np.linalg.norm(e_z) < IK_AXIS_EPS:
                        break
                    J = pin.computeFrameJacobian(
                        self._model, self._data, q, self._tcp_id, pin.LOCAL_WORLD_ALIGNED
                    )
                    Jp = J[:3, :5]
                    Jz = -pin.skew(z) @ J[3:, :5]
                    Je = np.vstack([Jp, axis_weight * Jz])
                    e = np.concatenate([e_p, axis_weight * e_z])
                    lam = 1e-4
                    dq = Je.T @ np.linalg.solve(Je @ Je.T + lam * np.eye(6), e)
                    step = np.linalg.norm(dq)
                    if step > 0.3:
                        dq *= 0.3 / step
                    q[:5] = np.clip(q[:5] + dq, self._q_lo, self._q_hi)
                # Polish: position only (orientation is only a preference on a 5-DoF arm).
                for _ in range(50):
                    T = self._forward_kinematics(q)
                    e_p = p_des - T.translation
                    if np.linalg.norm(e_p) < 1e-4:
                        break
                    Jp = pin.computeFrameJacobian(
                        self._model, self._data, q, self._tcp_id, pin.LOCAL_WORLD_ALIGNED
                    )[:3, :5]
                    dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + 1e-6 * np.eye(3), e_p)
                    q[:5] = np.clip(q[:5] + dq, self._q_lo, self._q_hi)
                T = self._forward_kinematics(q)
                pos_err = float(np.linalg.norm(p_des - T.translation))
                axis_err = float(np.linalg.norm(z_des - T.rotation[:, 2]))
                if pos_err < IK_POSITION_EPS and axis_err < IK_AXIS_EPS:
                    return q[:5].copy(), pos_err, axis_err
                # keep the best: position first, then approach axis
                if pos_err < best[1] - 1e-4 or (abs(pos_err - best[1]) <= 1e-4 and axis_err < best[2]):
                    best = (q[:5].copy(), pos_err, axis_err)
        return best

    # ------------------------------------------------------------------ motion

    def _start_motion(self, q_arm_target):
        """Single joint-space segment from the current position to q_arm_target."""
        self._start_trajectory([q_arm_target])

    def _start_trajectory(self, waypoints, min_segment_duration=None):
        """Piecewise-linear joint trajectory through the waypoints (list of q_arm).

        Each segment is timed by max_joint_speed, or by min_segment_duration (s) when
        that is longer (used by move_line to enforce a Cartesian speed).
        """
        with self._lock:
            q_prev = self._q[:5].copy()
            t = 0.0
            points = [(0.0, q_prev)]
            for q in waypoints:
                q = np.clip(np.asarray(q, dtype=float), self._q_lo, self._q_hi)
                dq_max = float(np.max(np.abs(q - q_prev)))
                dur = dq_max / self._max_joint_speed
                if min_segment_duration is not None:
                    dur = max(dur, min_segment_duration)
                t += dur
                points.append((t, q))
                q_prev = q
            if t < MIN_MOTION_DURATION:  # stretch a very short motion
                scale = MIN_MOTION_DURATION / max(t, 1e-6)
                points = [(ti * scale, qi) for ti, qi in points]
            self._traj_points = points
            self._traj_t0 = time.monotonic()

    def _motion_active(self) -> bool:
        with self._lock:
            return self._traj_points is not None

    def _wait_motion(self, timeout=20.0) -> bool:
        t_end = time.monotonic() + timeout
        while self._motion_active() and time.monotonic() < t_end:
            time.sleep(0.02)
        time.sleep(0.3)  # let the arm settle on the last command
        return not self._motion_active()

    def _control_step(self):
        with self._lock:
            if self._traj_points is None:
                return
            t = time.monotonic() - self._traj_t0
            points = self._traj_points
            if t >= points[-1][0]:
                q_cmd = points[-1][1]
                self._traj_points = None
            else:
                i = 1
                while points[i][0] < t:
                    i += 1
                (t0, q0), (t1, q1) = points[i - 1], points[i]
                s = (t - t0) / max(t1 - t0, 1e-6)
                q_cmd = q0 + s * (q1 - q0)
        self._send_joint_command(q_cmd)

    def _send_joint_command(self, q_arm, gripper_pct=None):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(ARM_JOINT_NAMES)
        msg.position = [float(v) for v in q_arm]
        if gripper_pct is not None:
            msg.name.append("gripper")
            msg.position.append(float(np.clip(gripper_pct, 0.0, 100.0)))
        self._pub_cmd.publish(msg)

    def set_gripper(self, pct: float):
        """Opens (100) / closes (0) the gripper, in %."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["gripper"]
        msg.position = [float(np.clip(pct, 0.0, 100.0))]
        self._pub_cmd.publish(msg)

    def go_to(self, position, approach_dir=(0.0, 0.0, -1.0), tolerance=POSITION_TOLERANCE, x_dir=None):
        """Blocking move of the TCP to a world position. Returns (ok, message).

        tolerance: accepted error between the target and the measured TCP (m). The
        simulated/real arm sags a little at full extension, so the pick & place steps
        use a looser value than the /go_to_target service.
        """
        q_arm, pos_err, axis_err = self._solve_ik(position, approach_dir, x_dir=x_dir)
        if q_arm is None or pos_err > POSITION_TOLERANCE:
            return False, f"IK failed: closest reachable point is {pos_err * 1000:.1f} mm away"
        tgt = PoseStamped()
        tgt.header.stamp = self.get_clock().now().to_msg()
        tgt.header.frame_id = "world"
        tgt.pose.position.x, tgt.pose.position.y, tgt.pose.position.z = map(float, position)
        tgt.pose.orientation.w = 1.0
        self._pub_target.publish(tgt)

        self._start_motion(q_arm)
        if not self._wait_motion():
            return False, "motion timeout"
        T = self._forward_kinematics(self._current_q())
        err = float(np.linalg.norm(T.translation - np.asarray(position)))
        ok = err <= tolerance
        return ok, (
            f"reached, error {err * 1000:.1f} mm (IK residual {pos_err * 1000:.1f} mm, "
            f"approach axis error {math.degrees(2 * math.asin(min(1.0, axis_err / 2))):.1f} deg)"
        )

    def move_line(self, target_position, approach_dir=(0.0, 0.0, -1.0), speed=LINE_DEFAULT_SPEED,
                  tolerance=POSITION_TOLERANCE, x_dir=None):
        """Blocking straight-line motion of the TCP to a world position. Returns (ok, message).

        The line is sampled every LINE_STEP; each waypoint is solved by IK seeded with the
        previous one (continuity), then executed as one joint trajectory at `speed` m/s.
        """
        q_now = self._current_q()
        p0 = self._forward_kinematics(q_now).translation
        p1 = np.asarray(target_position, dtype=float)
        dist = float(np.linalg.norm(p1 - p0))
        n = max(1, int(math.ceil(dist / LINE_STEP)))
        speed = speed if speed > 0 else LINE_DEFAULT_SPEED
        waypoints, q_seed = [], q_now
        for i in range(1, n + 1):
            p = p0 + (p1 - p0) * i / n
            q_arm, pos_err, _ = self._solve_ik(p, approach_dir, q_seed=q_seed, x_dir=x_dir)
            if q_arm is None or pos_err > POSITION_TOLERANCE:
                return False, f"IK failed at {i}/{n} ({p.round(3)}): {pos_err * 1000:.1f} mm away"
            waypoints.append(q_arm)
            q_seed = np.concatenate([q_arm, q_seed[5:]])
        self._start_trajectory(waypoints, min_segment_duration=(dist / n) / speed)
        if not self._wait_motion(timeout=dist / speed + 10.0):
            return False, "motion timeout"
        T = self._forward_kinematics(self._current_q())
        err = float(np.linalg.norm(T.translation - p1))
        return err <= tolerance, f"line done, error {err * 1000:.1f} mm ({n} waypoints)"

    # ---------------------------------------------------------------- services

    @staticmethod
    def _approach_from_msg(orientation):
        """Orientation of a request -> (approach direction = z axis, jaw direction = x axis).
        Identity / zero quaternion = default: come from above, jaw direction free."""
        quat = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
        if np.linalg.norm(quat) < 1e-6 or np.allclose(quat, [0, 0, 0, 1]):
            return (0.0, 0.0, -1.0), None
        R = pin.Quaternion(quat / np.linalg.norm(quat)).toRotationMatrix()
        return tuple(R[:, 2]), tuple(R[:, 0])

    def _srv_move_line(self, req: MoveLine.Request, res: MoveLine.Response):
        if not self._have_state:
            res.success, res.message = False, "no /joint_states yet"
            return res
        if req.target_pose.header.frame_id not in ("", "world", "base_link"):
            res.success, res.message = False, f"unsupported frame '{req.target_pose.header.frame_id}'"
            return res
        p = req.target_pose.pose.position
        approach, x_dir = self._approach_from_msg(req.target_pose.pose.orientation)
        res.success, res.message = self.move_line((p.x, p.y, p.z), approach, req.speed, x_dir=x_dir)
        self.get_logger().info(f"move_line -> ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}): {res.message}")
        return res

    def _srv_go_to_target(self, req: GoToTarget.Request, res: GoToTarget.Response):
        if not self._have_state:
            res.success, res.message = False, "no /joint_states yet"
            return res
        frame = req.target_pose.header.frame_id
        if frame not in ("", "world", "base_link"):
            res.success, res.message = False, f"unsupported frame '{frame}' (use world)"
            return res
        p = req.target_pose.pose.position
        approach, x_dir = self._approach_from_msg(req.target_pose.pose.orientation)
        self.get_logger().info(f"go_to_target ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) approach {np.round(approach, 2)}")
        res.success, res.message = self.go_to((p.x, p.y, p.z), approach, x_dir=x_dir)
        self.get_logger().info(res.message)
        return res

    def _srv_set_joint_positions(self, req: SetJointPositions.Request, res: SetJointPositions.Response):
        vals = list(req.positions)
        if len(vals) not in (5, 6):
            res.success, res.message = False, "expected 5 arm joints (deg) [+ gripper %]"
            return res
        q_arm = np.radians(vals[:5])
        clipped = np.clip(q_arm, self._q_lo, self._q_hi)
        if len(vals) == 6:
            self.set_gripper(vals[5])
        self._start_motion(clipped)
        done = self._wait_motion()
        note = "" if np.allclose(clipped, q_arm) else " (some joints clamped to their limits)"
        res.success = done
        res.message = ("done" if done else "motion timeout") + note
        return res


    # ------------------------------------------------------------ pick & place

    def _cb_target(self, name, msg: PoseStamped):
        p = msg.pose.position
        with self._lock:
            self._targets[name] = (np.array([p.x, p.y, p.z]), time.monotonic())

    def _latest(self, name, max_age=None):
        """Last known position of a target, or None if never seen / too old."""
        with self._lock:
            entry = self._targets.get(name)
        if entry is None:
            return None
        xyz, t = entry
        if max_age is not None and time.monotonic() - t > max_age:
            return None
        return xyz.copy()

    def _set_state(self, state: str):
        self._state = state
        self._pub_state.publish(String(data=state))
        self.get_logger().info(f"[{state}]")

    def _check_abort(self):
        if self._abort.is_set():
            raise RuntimeError("aborted by /brain/stop")

    def _move(self, position, what):
        self._check_abort()
        ok, msg = self.go_to(position, tolerance=self._motion_tolerance)
        if not ok:
            raise RuntimeError(f"{what}: {msg}")

    def _observe(self):
        """Arm out of the camera view, gripper open, wait for fresh detections."""
        self._set_state("OBSERVE")
        self.set_gripper(100.0)
        self._start_motion(np.radians(self._observe_joints_deg))
        self._wait_motion()
        t_end = time.monotonic() + 3.0
        while time.monotonic() < t_end:
            self._check_abort()
            if self._latest("object", 0.5) is not None and self._latest("box", 0.5) is not None:
                return
            time.sleep(0.1)

    def _pick_and_place(self) -> str:
        self._observe()
        obj = self._latest("object", self._max_target_age)
        box = self._latest("box")  # the box does not move: last known value is fine
        if obj is None:
            raise RuntimeError("object not visible (no fresh /object_position)")
        if box is None:
            raise RuntimeError("drop box never seen (/drop_box_position)")

        for attempt in range(1 + self._max_retries):
            # APPROACH above the object, re-plan while it keeps moving.
            self._set_state("APPROACH")
            for _ in range(4):
                self._move([obj[0], obj[1], self._approach_height], "approach")
                fresh = self._latest("object", self._max_target_age)
                if fresh is None or np.linalg.norm((fresh - obj)[:2]) < self._replan_distance:
                    break
                self.get_logger().warn(f"object moved by {np.linalg.norm((fresh - obj)[:2]) * 100:.1f} cm, re-planning")
                obj = fresh

            self._set_state("DESCEND")
            self._move([obj[0], obj[1], self._grasp_height], "descend")
            self._set_state("GRASP")
            self.set_gripper(0.0)
            time.sleep(0.8)
            self._set_state("LIFT")
            self._move([obj[0], obj[1], self._lift_height], "lift")
            time.sleep(0.5)

            # Grasp check: if the object is still seen where it was, we missed it.
            still_there = self._latest("object", 0.5)
            if still_there is not None and np.linalg.norm((still_there - obj)[:2]) < 0.03:
                self.get_logger().warn(f"grasp failed (object still on the table), attempt {attempt + 1}")
                self.set_gripper(100.0)
                time.sleep(0.5)
                obj = still_there
                continue

            self._set_state("TRANSPORT")
            self._move([box[0], box[1], self._lift_height], "transport")
            self._move([box[0], box[1], self._drop_height], "lower")
            self._set_state("RELEASE")
            self.set_gripper(100.0)
            time.sleep(0.8)
            self._set_state("RETREAT")
            self._move([box[0], box[1], self._lift_height], "retreat")
            self._observe()
            seen = self._latest("object", 1.0)
            where = "" if seen is None else f", object now seen at ({seen[0]:.2f}, {seen[1]:.2f})"
            return f"done in {attempt + 1} attempt(s){where}"
        raise RuntimeError(f"grasp failed {1 + self._max_retries} times")

    def _srv_pick_and_place(self, req, res: Trigger.Response):
        if not self._task_running.acquire(blocking=False):
            res.success, res.message = False, "a task is already running"
            return res
        self._abort.clear()
        try:
            res.message = self._pick_and_place()
            res.success = True
        except Exception as exc:  # noqa: BLE001
            res.success, res.message = False, str(exc)
            self.get_logger().error(f"pick & place failed: {exc}")
        finally:
            self._set_state("IDLE")
            self._task_running.release()
        return res

    def _srv_stop(self, req, res: Trigger.Response):
        self._abort.set()
        with self._lock:
            self._traj_points = None  # freeze where we are
        res.success, res.message = True, "stop requested"
        return res


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
