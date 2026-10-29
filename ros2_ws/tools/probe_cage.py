#!/usr/bin/env python3
"""probe_cage.py - geometry of the open gripper around the TCP (to cage the planchette).
Moves the TCP to (0.22, 0, 0.10) with the jaws along +x and the gripper open, then prints,
for the fixed and moving jaws, the extent (relative to the TCP, in the world frame) of the
mesh vertices that lie in the planchette height band once lowered.
    python3 /ros2_ws/tools/probe_cage.py [open_pct]"""
import sys

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

from so101_interfaces.srv import GoToTarget, SetJointPositions
from so101_sim import ASSETS_DIR, SO101Sim

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LIMITS = [(-1.91986, 1.91986), (-1.74533, 1.74533), (-1.69, 1.69), (-1.65806, 1.65806), (-2.74385, 2.84121)]


def main():
    open_pct = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    rclpy.init()
    node = Node("probe_cage")
    state = {}
    node.create_subscription(JointState, "/joint_states", lambda m: state.update(js=m), 10)
    node.create_subscription(PoseStamped, "/end_effector_pose", lambda m: state.update(ee=m), 10)
    cli_g = node.create_client(GoToTarget, "/go_to_target")
    cli_j = node.create_client(SetJointPositions, "/set_joint_positions")
    cli_g.wait_for_service(); cli_j.wait_for_service()
    req = SetJointPositions.Request(); req.positions = [0.0, -100.0, 40.0, -80.0, 0.0, open_pct]
    fut = cli_j.call_async(req); rclpy.spin_until_future_complete(node, fut, timeout_sec=30)

    req = GoToTarget.Request()
    req.target_pose.header.frame_id = "world"
    req.target_pose.pose.position.x, req.target_pose.pose.position.y, req.target_pose.pose.position.z = 0.22, 0.0, 0.10
    # z down, x = +x world: rotation of pi about x
    req.target_pose.pose.orientation.x, req.target_pose.pose.orientation.w = 1.0, 0.0
    fut = cli_g.call_async(req); rclpy.spin_until_future_complete(node, fut, timeout_sec=30)
    print("go_to:", fut.result().message)
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
    js = state["js"]; q = dict(zip(js.name, js.position))
    print("joints (deg):", {k: round(np.degrees(q[k]), 1) for k in JOINTS[:5]}, "gripper(rad):", round(q["gripper"], 3))
    for k, (lo, hi) in zip(JOINTS[:5], LIMITS):
        if abs(q[k] - lo) < 0.02 or abs(q[k] - hi) < 0.02:
            print(f"  !! {k} at its limit")

    sim = SO101Sim(scene_path=ASSETS_DIR / "scene_ouija.xml"); sim.connect()
    m, d = sim.model, sim.data
    for name in JOINTS:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        d.qpos[m.jnt_qposadr[jid]] = q[name]
    mujoco.mj_forward(m, d)
    ee = state["ee"].pose.position
    tcp = np.array([ee.x, ee.y, ee.z])
    print(f"TCP = {tcp.round(4)}")
    for body in ("gripper", "moving_jaw_so101_v1"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            verts = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            pts.append(d.geom_xpos[g] + verts @ d.geom_xmat[g].reshape(3, 3).T)
        pts = np.concatenate(pts) - tcp
        for lo, hi in ((-0.02, -0.012), (-0.012, -0.004), (-0.004, 0.004), (0.004, 0.012)):
            band = pts[(pts[:, 2] >= lo) & (pts[:, 2] < hi)]
            if len(band):
                print(f"  {body:22s} z in [{lo:+.3f},{hi:+.3f}] rel. TCP: x [{band[:,0].min():+.4f}, {band[:,0].max():+.4f}]"
                      f"  y [{band[:,1].min():+.4f}, {band[:,1].max():+.4f}]  ({len(band)} vertices)")
            else:
                print(f"  {body:22s} z in [{lo:+.3f},{hi:+.3f}] rel. TCP: nothing")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
