#!/usr/bin/env python3
"""hover_object.py - puts the gripper above the object seen by the perception (current /object_position).
    python3 /ros2_ws/tools/hover_object.py          # tip 4 cm above the centre of the cube
    python3 /ros2_ws/tools/hover_object.py 0.005    # tip 5 mm above the centre = grasp height
    python3 /ros2_ws/tools/hover_object.py 0.04 box # above the drop box instead
Prints the target and the answer of /go_to_target. Torque must be on (ARM_TORQUE=true)."""
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from so101_interfaces.srv import GoToTarget


def main():
    dz = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
    which = sys.argv[2] if len(sys.argv) > 2 else "object"
    topic = "/drop_box_position" if which == "box" else "/object_position"
    rclpy.init()
    node = Node("hover_object")
    box = {}
    node.create_subscription(PoseStamped, topic, lambda m: box.update(p=m.pose.position), 1)
    cli = node.create_client(GoToTarget, "/go_to_target")
    cli.wait_for_service()
    while "p" not in box and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.5)
        print(f"waiting for {topic}...") if "p" not in box else None
    for _ in range(5):  # take the latest value
        rclpy.spin_once(node, timeout_sec=0.2)
    p = box["p"]
    req = GoToTarget.Request()
    req.target_pose.header.frame_id = "world"
    req.target_pose.pose.position.x, req.target_pose.pose.position.y = p.x, p.y
    req.target_pose.pose.position.z = p.z + dz
    print(f"{which} seen at x={p.x:.3f} y={p.y:.3f} z={p.z:.3f} -> tip to z={p.z + dz:.3f}")
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut)
    print(fut.result().message)


if __name__ == "__main__":
    main()
