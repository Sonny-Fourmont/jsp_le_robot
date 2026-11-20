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
import os
import threading
import time
from collections import deque

import numpy as np
import pinocchio as pin
import rclpy
import rclpy.node
from geometry_msgs.msg import Point, PoseStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from sensor_msgs.msg import JointState
from so101_interfaces.srv import GoToTarget, MoveLine, SetJointPositions
from std_msgs.msg import String
from std_srvs.srv import Trigger
from pathlib import Path
from visualization_msgs.msg import Marker, MarkerArray

WORKSPACE_FILE = Path("/ros2_ws/calibration/workspace.yaml")


def load_table():
    """(x0, z0, slope) of the table plane in the robot frame on the real arm, (0, 0, 0) in simulation."""
    if os.environ.get("USE_SIM", "true").lower() == "true" or not WORKSPACE_FILE.is_file():
        return 0.0, 0.0, 0.0
    import yaml

    tb = (yaml.safe_load(WORKSPACE_FILE.read_text()) or {}).get("table", {})
    return float(tb.get("x0", 0.0)), float(tb.get("z0", 0.0)), float(tb.get("slope", 0.0))

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
GRIPPER_URDF_RANGE = (-0.174533, 1.74533)  # gripper joint in /joint_states: closed, open (rad)
MOTION_HOLD = 0.3  # s, the final command of a trajectory is re-sent during this time
MIN_MOTION_DURATION = 0.3  # s
LINE_STEP = 0.01  # m, waypoint spacing of /move_line
LINE_DEFAULT_SPEED = 0.05  # m/s
STRAIGHT_SPEED = 0.12  # m/s, pick & place descend / lift (go_to straight=True)
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
        # Wrist roll kept within +/- wrist_roll_limit_deg: the jaws are symmetric, so any jaw
        # direction is reachable within +/- 90 deg, and a cable (wrist camera) runs to the gripper
        # on the real arm: it got unplugged at 159 deg.
        roll_lim = math.radians(float(self.declare_parameter("wrist_roll_limit_deg", 90.0).value))
        self._q_lo[4], self._q_hi[4] = max(self._q_lo[4], -roll_lim), min(self._q_hi[4], roll_lim)

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
        # RViz: FK position as text + sphere, trace of the TCP, current IK target (see _publish_end_effector_pose)
        self._pub_markers = self.create_publisher(MarkerArray, "/end_effector_markers", 5)
        self._tcp_trace = deque()  # (time, position) of the TCP over the last TRACE_SECONDS
        self._last_target = None  # (time, position) of the last IK target

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
        # Observe / parking pose: arm raised, gripper horizontal 30 cm up and 9 cm in front of the
        # base, out of the camera's view of the workspace. (The former folded pose had the TCP
        # 2.7 cm above the table at r = 17 cm: it swept away cubes placed near the base.)
        self._observe_joints_deg = pp("observe_joints_deg", [0.0, -90.0, -5.0, 90.0, 0.0]).value
        # Real arm: table plane z(x) from calibration/workspace.yaml, all target heights are above it
        self._table_x0, self._table_z0, self._table_slope = load_table()
        if self._table_z0 or self._table_slope:
            self.get_logger().info(f"table plane z(x) = {self._table_z0:.4f} + {self._table_slope:.4f} (x - {self._table_x0:.4f})")
        self._approach_height = pp("approach_height", 0.12).value
        # pre-grasp: highest of these where the gripper can be vertical (the descend starts there)
        self._pregrasp_heights = (self._approach_height, 0.10, 0.08, 0.06)
        # TCP height for the grasp: 3 cm = the fixed finger (tip 6 mm below the TCP) goes through the
        # diamond hole of the cube's top face down to mid-height, inside the hollow cube (see grasp_offset).
        self._grasp_height = pp("grasp_height", 0.03).value
        self._lift_height = pp("lift_height", 0.15).value
        self._drop_height = pp("drop_height", 0.10).value
        self._max_target_age = pp("max_target_age", 2.0).value  # s, freshness of a detection
        self._max_retries = pp("max_grasp_retries", 2).value
        self._replan_distance = pp("replan_distance", 0.02).value  # m, object moved -> re-approach
        self._motion_tolerance = pp("motion_tolerance", 0.03).value  # m, accepted error of a step
        # Closed-loop correction of go_to on the measured FK error: the real arm sags under
        # gravity (9-15 mm measured with P=16 at 9 V), so after a motion the TCP is below /
        # short of the target. We re-command the target shifted by the measured error, up to
        # `goto_corrections` times, when the error exceeds `goto_correction_min_error`.
        # In simulation the first move lands within ~1 mm, so no correction is triggered.
        self._goto_corrections = int(pp("goto_corrections", 2).value)
        self._goto_correction_min = float(pp("goto_correction_min_error", 0.003).value)  # m
        self._goto_correction_gain = float(pp("goto_correction_gain", 0.8).value)  # < 1: no overshoot on a noisy measure
        # grasp check: jaws closed below this opening (%) = nothing between them (5 cm cube ~ 35 %)
        # Gripper opening below which the jaws closed on nothing. Hooked through the cube's holes
        # (see grasp_offset) the jaws still close to 2-8 %, so the threshold is low; the main grasp
        # check is the perception (object still seen where it was).
        self._grasp_min_gripper_pct = float(pp("grasp_min_gripper_pct", 1.0).value)
        # Where to put the TCP relative to the object centre, along the jaw axis +x_tcp (from the
        # moving jaw towards the fixed finger). The TCP (gripper_frame_link) is the inner face of
        # the FIXED finger, which spans x_tcp 0..+14 mm. The 5 cm hollow cube has a 38 mm diamond
        # hole in each face: with the TCP 7 mm before the centre the finger (x_tcp 0..14) is centred
        # in the top hole and goes down inside the cube; the moving jaw then pushes the cube until
        # its far wall meets the finger, which pokes into that wall's side hole: the cube is hooked,
        # it cannot slide out of the jaws (a plain pinch on the outer faces slips: the moving jaw
        # is a lever that presses the top edge downwards). Tolerance about +-12 mm thanks to the
        # hole's slanted rim that centres the cube under the finger.
        # For a plain object use +(half width + 3 mm): the TCP just beyond the near face.
        self._grasp_offset = float(pp("grasp_offset", -0.007).value)

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
        """FK of the current joint state -> /end_effector_pose (PoseStamped) + RViz markers.

        The TF tree shown by RViz (robot_state_publisher, same URDF) is an independent computation
        of the same chain: the sphere/text published here lands exactly on the gripper_frame_link
        axes, which is the check to show (ros2 run tf2_ros tf2_echo world gripper_frame_link vs
        ros2 topic echo /end_effector_pose)."""
        if not self._have_state:
            return
        T = self._forward_kinematics(self._current_q())
        self._pub_pose.publish(self._se3_to_msg(T))
        self._pub_markers.publish(self._end_effector_markers(T.translation))

    TRACE_SECONDS = 20.0

    def _end_effector_markers(self, p) -> MarkerArray:
        now = time.monotonic()
        stamp = self.get_clock().now().to_msg()

        def marker(ns, mid, mtype, rgba, scale):
            m = Marker()
            m.header.stamp, m.header.frame_id = stamp, "world"
            m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = scale
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.lifetime.sec = 2
            return m

        out = MarkerArray()
        sphere = marker("tcp_fk", 0, Marker.SPHERE, (1.0, 0.85, 0.1, 0.6), (0.024, 0.024, 0.024))
        sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = map(float, p)
        out.markers.append(sphere)
        label = marker("tcp_fk", 1, Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.016))
        label.pose.position.x, label.pose.position.y, label.pose.position.z = float(p[0]), float(p[1]) + 0.09, float(p[2]) + 0.04
        # no spaces: RViz (MovableText) draws them several decimetres wide
        label.text = f"TCP=gripper_frame_link\nFK:x={p[0]:+.3f},y={p[1]:+.3f},z={p[2]:+.3f}"
        out.markers.append(label)

        # trace of the TCP: a point every 2 mm of travel, kept TRACE_SECONDS
        if not self._tcp_trace or np.linalg.norm(self._tcp_trace[-1][1] - p) > 0.002:
            self._tcp_trace.append((now, np.asarray(p, dtype=float).copy()))
        while self._tcp_trace and now - self._tcp_trace[0][0] > self.TRACE_SECONDS:
            self._tcp_trace.popleft()
        trace = marker("tcp_trace", 2, Marker.LINE_STRIP, (0.2, 0.9, 1.0, 0.9), (0.003, 0.0, 0.0))
        if len(self._tcp_trace) >= 2:
            trace.points = [Point(x=float(q[0]), y=float(q[1]), z=float(q[2])) for _, q in self._tcp_trace]
        else:
            trace.action = Marker.DELETE
        out.markers.append(trace)

        # last IK target (where the brain asked the TCP to go), magenta, 5 s after the request
        target = marker("ik_target", 3, Marker.SPHERE, (1.0, 0.2, 0.9, 0.8), (0.016, 0.016, 0.016))
        if self._last_target is not None and now - self._last_target[0] < 5.0:
            t = self._last_target[1]
            target.pose.position.x, target.pose.position.y, target.pose.position.z = map(float, t)
        else:
            target.action = Marker.DELETE
        out.markers.append(target)
        return out

    def _se3_to_msg(self, T: pin.SE3) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, T.translation)
        qx, qy, qz, qw = pin.Quaternion(T.rotation).coeffs()
        msg.pose.orientation.x, msg.pose.orientation.y = float(qx), float(qy)
        msg.pose.orientation.z, msg.pose.orientation.w = float(qz), float(qw)
        return msg

    def _solve_ik(self, target_position, approach_dir=(0.0, 0.0, -1.0), q_seed=None, x_dir=None, local=False,
                  symmetric=True):
        """Damped least squares IK on the 5 arm joints.

        Tasks: TCP position (3) + TCP z axis aligned with approach_dir (2 effective).
        x_dir (optional, default world x): desired direction of the TCP x axis (the jaw axis),
        obtained afterwards with the wrist roll, which is free when the gripper points down.
        local: solve from q_seed only (no multi-start), to stay in the same arm configuration
        when re-targeting a few millimetres away (go_to corrections).
        symmetric: x_dir and -x_dir are equivalent (default); False keeps the sign, needed when the
        side of the moving jaw matters (grasp).
        Returns (q_arm, pos_error_m, axis_error) for the best solution found.
        """
        q_arm, pos_err, axis_err = self._solve_ik_position_axis(target_position, approach_dir, q_seed, local)
        if x_dir is None:
            x_dir = (1.0, 0.0, 0.0)  # default: jaws along the robot x axis (roll near 0, never left random)
        if q_arm is not None:
            p_des = np.asarray(target_position, dtype=float)
            z_des = np.asarray(approach_dir, dtype=float)
            z_des = z_des / np.linalg.norm(z_des)
            # The TCP is 7.9 mm off the roll axis (gripper_frame_joint), so turning the roll moves
            # it by up to 8 mm: align the roll, then re-polish the position with the roll locked.
            # Twice, because the polish turns the TCP x axis a little.
            for _ in range(2):
                q_arm = self._align_roll(q_arm, x_dir, symmetric)
                q_arm, pos_err, axis_err = self._polish_position(q_arm, p_des, z_des)
        return q_arm, pos_err, axis_err

    def _polish_position(self, q_arm, p_des, z_des, axis_weight=0.05):
        """Re-solves position + approach axis on the first 4 joints (wrist roll locked), then
        polishes the position only, like _solve_ik_position_axis. Keeping the axis task here
        matters: a position-only solve would tilt the gripper to absorb the roll offset.
        Returns (q_arm, pos_error_m, axis_error) measured on the result."""
        q = np.concatenate([q_arm, [0.0]])
        for _ in range(100):
            T = self._forward_kinematics(q)
            z = T.rotation[:, 2]
            e_p = p_des - T.translation
            e_z = z_des - z
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_z) < IK_AXIS_EPS:
                break
            J = pin.computeFrameJacobian(self._model, self._data, q, self._tcp_id, pin.LOCAL_WORLD_ALIGNED)
            Je = np.vstack([J[:3, :4], axis_weight * (-pin.skew(z) @ J[3:, :4])])
            e = np.concatenate([e_p, axis_weight * e_z])
            dq = Je.T @ np.linalg.solve(Je @ Je.T + 1e-4 * np.eye(6), e)
            step = np.linalg.norm(dq)
            if step > 0.3:
                dq *= 0.3 / step
            q[:4] = np.clip(q[:4] + dq, self._q_lo[:4], self._q_hi[:4])
        for _ in range(50):
            T = self._forward_kinematics(q)
            e_p = p_des - T.translation
            if np.linalg.norm(e_p) < 1e-4:
                break
            Jp = pin.computeFrameJacobian(
                self._model, self._data, q, self._tcp_id, pin.LOCAL_WORLD_ALIGNED
            )[:3, :4]
            dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + 1e-6 * np.eye(3), e_p)
            q[:4] = np.clip(q[:4] + dq, self._q_lo[:4], self._q_hi[:4])
        T = self._forward_kinematics(q)
        pos_err = float(np.linalg.norm(p_des - T.translation))
        axis_err = float(np.linalg.norm(z_des - T.rotation[:, 2]))
        return q[:5].copy(), pos_err, axis_err

    def _align_roll(self, q_arm, x_dir, symmetric=True):
        """Turns the wrist roll so that the TCP x axis (jaw axis) points along x_dir
        (projected on the plane perpendicular to the TCP z axis). symmetric: an angle differing
        by 180 degrees is equivalent, the smaller rotation is used (jaw axis only); otherwise the
        sign is kept (the moving jaw is on the -x_tcp side)."""
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
            if symmetric:
                return (e + math.pi / 2) % math.pi - math.pi / 2  # jaw axis only
            return (e + math.pi) % (2 * math.pi) - math.pi

        # The roll axis may be +z or -z of the TCP (try both signs), and the roll has limits: when
        # the nearest alignment is clipped, the one 180 deg away (same jaw axis) usually fits. Keep
        # the candidate with the smallest residual, then the smallest roll change.
        lo, hi = self._q_lo[4], self._q_hi[4]
        for _ in range(3):
            e = error(q)
            if abs(e) < 1e-3:
                break
            q4 = q[4]
            turns = (0.0, math.pi, -math.pi) if symmetric else (0.0,)
            # residuals within 1 deg count as equal: then the smallest roll change wins (otherwise a
            # 180 deg turn can win on a 0.01 deg difference and flip the side of the moving jaw)
            best = (round(abs(e) / 0.017), 0.0, q4)
            for sign in (1.0, -1.0):
                for turn in turns:
                    cand = float(np.clip(q4 + sign * e + turn, lo, hi))
                    q[4] = cand
                    best = min(best, (round(abs(error(q)) / 0.017), abs(cand - q4), cand))
            q[4] = best[2]
        return q[:5].copy()

    def _solve_ik_position_axis(self, target_position, approach_dir, q_seed, local=False):
        p_des = np.asarray(target_position, dtype=float)
        z_des = np.asarray(approach_dir, dtype=float)
        z_des = z_des / np.linalg.norm(z_des)
        q_now = self._current_q() if q_seed is None else np.asarray(q_seed, dtype=float)
        if q_now.shape[0] == 5:  # arm joints only: add the gripper (not part of the TCP chain)
            q_now = np.concatenate([q_now, [0.0]])

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
        if local:
            seeds = seeds[:1]  # current configuration only
        nj = 4 if local else 5  # local: wrist roll locked too (same jaw axis, same side of the moving jaw)

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
                    Jp = J[:3, :nj]
                    Jz = -pin.skew(z) @ J[3:, :nj]
                    Je = np.vstack([Jp, axis_weight * Jz])
                    e = np.concatenate([e_p, axis_weight * e_z])
                    lam = 1e-4
                    dq = Je.T @ np.linalg.solve(Je @ Je.T + lam * np.eye(6), e)
                    step = np.linalg.norm(dq)
                    if step > 0.3:
                        dq *= 0.3 / step
                    q[:nj] = np.clip(q[:nj] + dq, self._q_lo[:nj], self._q_hi[:nj])
                # Polish: position only (orientation is only a preference on a 5-DoF arm).
                for _ in range(50):
                    T = self._forward_kinematics(q)
                    e_p = p_des - T.translation
                    if np.linalg.norm(e_p) < 1e-4:
                        break
                    Jp = pin.computeFrameJacobian(
                        self._model, self._data, q, self._tcp_id, pin.LOCAL_WORLD_ALIGNED
                    )[:3, :nj]
                    dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + 1e-6 * np.eye(3), e_p)
                    q[:nj] = np.clip(q[:nj] + dq, self._q_lo[:nj], self._q_hi[:nj])
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
        return not self._motion_active()

    def _jaw_axis(self, q_arm):
        """Jaw axis (+x of the TCP frame) of a configuration, projected on the table, unit, world."""
        x = self._forward_kinematics(np.concatenate([q_arm, [0.0]])).rotation[:, 0]
        x = np.array([x[0], x[1], 0.0])
        n = np.linalg.norm(x)
        return x / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

    def _gripper_pct(self) -> float:
        """Measured gripper opening in % (0 closed, 100 open), from /joint_states (URDF rad)."""
        lo, hi = GRIPPER_URDF_RANGE
        return float(np.clip((self._current_q()[5] - lo) / (hi - lo), 0.0, 1.0) * 100.0)

    def _wait_settled(self, timeout=1.0, eps=0.003):
        """Waits until the measured joints stop moving (|dq| < eps rad over 0.1 s, eps above the
        0.0015 rad tick of the real servos), timeout max."""
        t_end = time.monotonic() + timeout
        q0 = self._current_q()[:5]
        while time.monotonic() < t_end:
            time.sleep(0.1)
            q1 = self._current_q()[:5]
            if float(np.max(np.abs(q1 - q0))) < eps:
                return True
            q0 = q1
        return False

    def _control_step(self):
        with self._lock:
            if self._traj_points is None:
                return
            t = time.monotonic() - self._traj_t0
            points = self._traj_points
            if t >= points[-1][0] + MOTION_HOLD:
                q_cmd = points[-1][1]
                self._traj_points = None
            elif t >= points[-1][0]:
                # Hold: the final command is re-sent for MOTION_HOLD s. The driver caps every
                # command to max_step from the measured position (log "capped"), so a command that
                # was sent only once after a timer hiccup would leave the arm short of the target.
                q_cmd = points[-1][1]
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

    def go_to(self, position, approach_dir=(0.0, 0.0, -1.0), tolerance=POSITION_TOLERANCE, x_dir=None,
              correct=True, local=False, jaw_offset=0.0, straight=False):
        """Blocking move of the TCP to a world position. Returns (ok, message).

        correct: closed-loop corrections on the measured FK error (goto_corrections), for the
        steps where millimetres matter (grasp, /go_to_target); off for the coarse transit steps.
        local: solve the IK from the current configuration only (same posture and wrist roll as
        now), with the multi-start solve as fallback; used for the short pick & place steps.
        jaw_offset: `position` is the centre of an object to take between the jaws: the TCP
        (inner face of the fixed jaw) is placed jaw_offset beyond it along the jaw axis +x_tcp.
        straight: Cartesian straight line (waypoints every LINE_STEP, same jaw axis) instead of
        one joint-space segment, whose curved TCP path can clip an object a few mm away.

        tolerance: accepted error between the target and the measured TCP (m). The
        simulated/real arm sags a little at full extension, so the pick & place steps
        use a looser value than the /go_to_target service.
        """
        target = np.asarray(position, dtype=float)
        symmetric = True
        if local:
            # keep the jaw axis AND its sign (the side of the moving jaw) of the current posture
            if x_dir is None:
                x_dir = self._jaw_axis(self._current_q()[:5])
            symmetric = False
        q_arm, pos_err, axis_err = self._solve_ik(target, approach_dir, x_dir=x_dir, local=local, symmetric=symmetric)
        if local and (q_arm is None or pos_err > POSITION_TOLERANCE):  # multi-start fallback
            q_arm, pos_err, axis_err = self._solve_ik(target, approach_dir, x_dir=x_dir, symmetric=symmetric)
        if q_arm is None or pos_err > POSITION_TOLERANCE:
            return False, f"IK failed: closest reachable point is {pos_err * 1000:.1f} mm away"
        if jaw_offset:
            # Shift the TCP along the jaw axis of the solution found, re-solving from that solution
            # (same wrist roll, so the jaw axis keeps its direction and the moving jaw its side).
            for _ in range(2):
                jaw = self._jaw_axis(q_arm)
                shifted = target + jaw_offset * jaw
                q2, pos_err2, axis_err2 = self._solve_ik(shifted, approach_dir, q_seed=q_arm, x_dir=jaw, local=True,
                                                        symmetric=False)
                if q2 is None or pos_err2 > POSITION_TOLERANCE:  # local solve stuck: multi-start
                    q2, pos_err2, axis_err2 = self._solve_ik(shifted, approach_dir, x_dir=jaw, symmetric=False)
                if q2 is None or pos_err2 > POSITION_TOLERANCE:
                    return False, f"IK failed: grasp point {pos_err2 * 1000:.1f} mm out of reach"
                q_arm, pos_err, axis_err = q2, pos_err2, axis_err2
                if np.dot(self._jaw_axis(q_arm), jaw) > 0.9:
                    break
            self.get_logger().info(
                f"jaw offset {jaw_offset * 1000:.0f} mm along ({jaw[0]:+.2f}, {jaw[1]:+.2f}): "
                f"object ({target[0]:.3f}, {target[1]:.3f}) -> TCP ({shifted[0]:.3f}, {shifted[1]:.3f})")
            target = shifted
        tgt = PoseStamped()
        tgt.header.stamp = self.get_clock().now().to_msg()
        tgt.header.frame_id = "world"
        tgt.pose.position.x, tgt.pose.position.y, tgt.pose.position.z = map(float, target)
        tgt.pose.orientation.w = 1.0
        self._pub_target.publish(tgt)
        self._last_target = (time.monotonic(), np.asarray(target, dtype=float).copy())

        commanded = target.copy()  # what we ask the IK for; drifts away from `target` by the measured sag
        corrections = 0
        best = None  # (error, q_arm) of the best command so far
        path = self._straight_path(target, q_arm, approach_dir) if straight else None
        while True:
            if path:
                self._start_trajectory(path, min_segment_duration=LINE_STEP / STRAIGHT_SPEED)
                path = None
            else:
                self._start_motion(q_arm)
            if not self._wait_motion():
                return False, "motion timeout"
            if correct:
                self._wait_settled()
            T = self._forward_kinematics(self._current_q())
            err_vec = target - T.translation
            err = float(np.linalg.norm(err_vec))
            if best is not None and err > best[0] + 1e-3:
                # Worse than before (noisy measure, contact...): back to the best command, stop.
                self.get_logger().warning(
                    f"go_to correction {corrections}: error {err * 1000:.1f} mm, worse than "
                    f"{best[0] * 1000:.1f} mm before: reverting to the previous command")
                q_arm = best[1]
                self._start_motion(q_arm)
                self._wait_motion()
                self._wait_settled()
                T = self._forward_kinematics(self._current_q())
                err_vec = target - T.translation
                err = float(np.linalg.norm(err_vec))
                break
            best = (err, q_arm)
            if not correct or err <= self._goto_correction_min or corrections >= self._goto_corrections:
                break
            # Retarget by the measured error (the arm lands at commanded - sag, so ask for target + sag),
            # solving the IK from the current configuration only (same posture, same sag).
            shift = self._goto_correction_gain * err_vec
            commanded = commanded + shift
            q_new, pos_err2, axis_err2 = self._solve_ik(commanded, approach_dir, q_seed=self._current_q(),
                                                        x_dir=x_dir, local=True, symmetric=symmetric)
            if q_new is None or pos_err2 > POSITION_TOLERANCE:
                self.get_logger().warning(
                    f"go_to correction {corrections + 1}: error {err * 1000:.1f} mm but the shifted "
                    f"target is out of reach ({pos_err2 * 1000:.1f} mm away), keeping the current pose")
                break
            corrections += 1
            self.get_logger().info(
                f"go_to correction {corrections}: measured error {err * 1000:.1f} mm, retarget by "
                f"({shift[0] * 1000:+.1f}, {shift[1] * 1000:+.1f}, {shift[2] * 1000:+.1f}) mm")
            q_arm, axis_err = q_new, axis_err2
        ok = err <= tolerance
        return ok, (
            f"reached, error {err * 1000:.1f} mm after {corrections} correction(s) "
            f"(IK residual {pos_err * 1000:.1f} mm, "
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

    def _straight_path(self, target, q_end, approach_dir):
        """Joint waypoints along the straight TCP line from the current position to `target`,
        ending exactly at q_end; same jaw axis and side as q_end all along. None if the line
        cannot be solved (the caller then moves in joint space)."""
        q_now = self._current_q()
        p0 = self._forward_kinematics(q_now).translation
        p1 = np.asarray(target, dtype=float)
        dist = float(np.linalg.norm(p1 - p0))
        n = int(math.ceil(dist / LINE_STEP))
        if n < 2:
            return None
        jaw = self._jaw_axis(q_end)
        waypoints, q_seed = [], q_now
        for i in range(1, n):
            p = p0 + (p1 - p0) * i / n
            q_arm, pos_err, _ = self._solve_ik(p, approach_dir, q_seed=q_seed, x_dir=jaw, local=True,
                                               symmetric=False)
            if q_arm is None or pos_err > POSITION_TOLERANCE:
                self.get_logger().warning(f"straight path: waypoint {i}/{n} unreachable, moving in joint space")
                return None
            waypoints.append(q_arm)
            q_seed = np.concatenate([q_arm, q_seed[5:]])
        waypoints.append(np.asarray(q_end, dtype=float))
        return waypoints

    def _move(self, position, what, correct=False, local=False, jaw_offset=0.0, straight=False):
        self._check_abort()
        ok, msg = self.go_to(position, tolerance=self._motion_tolerance, correct=correct, local=local,
                             jaw_offset=jaw_offset, straight=straight)
        if not ok:
            raise RuntimeError(f"{what}: {msg}")

    def _observe(self):
        """Arm out of the camera view, gripper open, wait for fresh detections."""
        self._set_state("OBSERVE")
        self.set_gripper(100.0)
        self._start_motion(np.radians(self._observe_joints_deg))
        self._wait_motion()
        # Wait for a *stable* detection: while the arm leaves the camera view it can hide part of
        # the cube, and the centroid of a half-hidden cube is off by 1-2 cm (seen in simulation:
        # the gripper then lands on the cube and pushes it away). 3 consecutive samples within 5 mm.
        t_end = time.monotonic() + 3.0
        hist, last_stamp = [], None
        while time.monotonic() < t_end:
            self._check_abort()
            with self._lock:
                entry = self._targets.get("object")
            if entry is not None and time.monotonic() - entry[1] < 0.5 and self._latest("box", 0.5) is not None:
                if entry[1] != last_stamp:  # a new sample (perception at 10 Hz)
                    hist.append(entry[0].copy())
                    last_stamp = entry[1]
                if len(hist) >= 3 and max(np.linalg.norm((h - hist[-1])[:2]) for h in hist[-3:]) < 0.005:
                    return
            time.sleep(0.05)
        if hist:
            self.get_logger().warn("object detection still moving after 3 s, using the last one")

    def _tz(self, x) -> float:
        """Height of the table at abscissa x (0 in simulation); target heights are above the table.
        calibration/workspace.yaml is re-read when it changes (no restart needed to tune it)."""
        try:
            mtime = WORKSPACE_FILE.stat().st_mtime
        except OSError:
            mtime = None
        if mtime != getattr(self, "_table_mtime", None):
            self._table_mtime = mtime
            self._table_x0, self._table_z0, self._table_slope = load_table()
            self.get_logger().info(f"table plane z(x) = {self._table_z0:.4f} + {self._table_slope:.4f} (x - {self._table_x0:.4f})")
        return self._table_z0 + self._table_slope * (x - self._table_x0)

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
                # Transit at approach_height (the gripper may be tilted there: at 12 cm the wrist
                # cannot point straight down), then a straight move to the pre-grasp pose with a
                # vertical gripper, so that the descend is a pure vertical translation and the
                # fixed finger enters the object's hole instead of landing on it.
                # All heights are above the table plane (self._tz, 0 in simulation).
                tz = self._tz(obj[0])
                self._move([obj[0], obj[1], tz + self._approach_height], "approach", jaw_offset=self._grasp_offset)
                for z in self._pregrasp_heights:
                    q_try, err_try, axis_try = self._solve_ik([obj[0], obj[1], tz + z])
                    if q_try is not None and err_try <= POSITION_TOLERANCE and axis_try < IK_AXIS_EPS:
                        if z < self._approach_height:
                            self._move([obj[0], obj[1], tz + z], "pre-grasp", local=True, straight=True,
                                       jaw_offset=self._grasp_offset)
                        break
                else:
                    self.get_logger().warn("no height with a vertical gripper above the object, descending tilted")
                fresh = self._latest("object", self._max_target_age)
                if fresh is None or np.linalg.norm((fresh - obj)[:2]) < self._replan_distance:
                    break
                self.get_logger().warn(f"object moved by {np.linalg.norm((fresh - obj)[:2]) * 100:.1f} cm, re-planning")
                obj = fresh

            self._set_state("DESCEND")
            # local: same posture and roll as the approach, so the TCP goes straight down
            tz = self._tz(obj[0])
            try:
                self._move([obj[0], obj[1], tz + self._grasp_height], "descend", correct=True, local=True,
                           straight=True, jaw_offset=self._grasp_offset)  # the grasp needs the mm
            except RuntimeError as exc:
                if "reached, error" not in str(exc) or attempt >= self._max_retries:
                    raise
                # Blocked well above the target: the fixed finger landed on the object instead of in
                # its hole (perception off, or tilted gripper near the base). Back up and retry.
                self.get_logger().warn(f"descend blocked ({exc}), attempt {attempt + 1}: backing up to retry")
                self._move([obj[0], obj[1], tz + self._lift_height], "back up", local=True,
                           jaw_offset=self._grasp_offset, straight=True)
                self._observe()
                fresh = self._latest("object", self._max_target_age)
                if fresh is not None:
                    obj = fresh
                continue
            self._set_state("GRASP")
            self.set_gripper(0.0)
            time.sleep(0.8)
            self._set_state("LIFT")
            self._move([obj[0], obj[1], tz + self._lift_height], "lift", local=True, jaw_offset=self._grasp_offset,
                       straight=True)
            time.sleep(0.5)

            # Grasp check: jaws fully closed = nothing between them; object still seen where it
            # was = we missed it (or pushed it: the fresh position is reused for the retry).
            still_there = self._latest("object", 0.5)
            missed = still_there is not None and np.linalg.norm((still_there - obj)[:2]) < 0.03
            if self._gripper_pct() < self._grasp_min_gripper_pct or missed:
                why = "jaws closed on nothing" if not missed else "object still on the table"
                self.get_logger().warn(
                    f"grasp failed ({why}, gripper {self._gripper_pct():.0f} %), attempt {attempt + 1}")
                self.set_gripper(100.0)
                time.sleep(0.5)
                if still_there is not None:
                    obj = still_there
                self._observe()  # fresh, stable detection for the retry
                fresh = self._latest("object", self._max_target_age)
                if fresh is not None:
                    obj = fresh
                continue

            self._set_state("TRANSPORT")
            # same jaw offset over the box, so that the object (not the TCP) ends at its centre
            tzb = self._tz(box[0])
            self._move([box[0], box[1], tzb + self._lift_height], "transport", local=True, jaw_offset=self._grasp_offset)
            self._move([box[0], box[1], tzb + self._drop_height], "lower", local=True, jaw_offset=self._grasp_offset)
            self._set_state("RELEASE")
            self.set_gripper(100.0)
            time.sleep(0.8)
            self._set_state("RETREAT")
            self._move([box[0], box[1], tzb + self._lift_height], "retreat", local=True, jaw_offset=self._grasp_offset)
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
