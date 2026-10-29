#!/usr/bin/env python3
"""probe_tool_geometry.py - where is the TCP relative to the gripper's lowest point?

Moves the TCP (pointing down) to a few points, then rebuilds the arm pose in a private
MuJoCo model and prints the lowest vertex of the gripper meshes vs the TCP height.
    python3 /ros2_ws/tools/probe_tool_geometry.py
"""
import math
import sys

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

from so101_interfaces.srv import GoToTarget
from so101_sim import ASSETS_DIR, SO101Sim

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    rclpy.init()
    node = Node("probe_tool_geometry")
    state = {}
    node.create_subscription(JointState, "/joint_states", lambda m: state.update(js=m), 10)
    node.create_subscription(PoseStamped, "/end_effector_pose", lambda m: state.update(ee=m), 10)
    cli = node.create_client(GoToTarget, "/go_to_target")
    cli.wait_for_service()
    sim = SO101Sim(scene_path=ASSETS_DIR / "scene_ouija.xml")
    sim.connect()
    m, d = sim.model, sim.data
    gripper_bodies = [i for i in range(m.nbody) if any(k in m.body(i).name for k in ("gripper", "jaw", "wrist"))]
    print("gripper bodies:", [m.body(i).name for i in gripper_bodies])

    for x, y, z in [(0.20, 0.0, 0.06), (0.20, 0.0, 0.03), (0.28, 0.0, 0.03), (0.16, 0.12, 0.03)]:
        req = GoToTarget.Request()
        req.target_pose.header.frame_id = "world"
        req.target_pose.pose.position.x, req.target_pose.pose.position.y, req.target_pose.pose.position.z = x, y, z
        req.target_pose.pose.orientation.w = 1.0
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=30.0)
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.1)
        js = state["js"]
        for name, pos in zip(js.name, js.position):
            if name in JOINTS:
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0:
                    d.qpos[m.jnt_qposadr[jid]] = pos
        mujoco.mj_forward(m, d)
        lowest = (np.inf, "")
        for g in range(m.ngeom):
            if m.geom_bodyid[g] not in gripper_bodies or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            verts = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            world = d.geom_xpos[g] + verts @ d.geom_xmat[g].reshape(3, 3).T
            k = int(np.argmin(world[:, 2]))
            if world[k, 2] < lowest[0]:
                lowest = (world[k, 2], f"{m.body(m.geom_bodyid[g]).name} at xy=({world[k,0]:.3f},{world[k,1]:.3f})")
        ee = state["ee"].pose.position
        print(f"target z={z:.3f}: {fut.result().message}")
        print(f"   TCP (FK) = ({ee.x:.3f}, {ee.y:.3f}, {ee.z:.3f}); lowest gripper vertex z={lowest[0]:.4f} ({lowest[1]})"
              f" -> {(ee.z - lowest[0]) * 1000:.1f} mm below the TCP")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
