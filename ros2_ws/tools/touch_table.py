#!/usr/bin/env python3
"""touch_table.py - measures the height of the table in the robot frame WITH THE GRIPPER VERTICAL
(the posture of the grasp). The tip goes above (x, y), then comes down 5 mm at each ENTER; type s +
ENTER when the tip touches the paper, q to cancel. Clear the area (no cube, no bowl).
    python3 /ros2_ws/tools/touch_table.py 0.25 0.0
Prints the z reached: it is the table height at this x for a vertical gripper."""
import sys

import rclpy
from rclpy.node import Node

from so101_interfaces.srv import GoToTarget


def main():
    x, y = float(sys.argv[1]), float(sys.argv[2])
    z = float(sys.argv[3]) if len(sys.argv) > 3 else 0.06
    step = float(sys.argv[4]) if len(sys.argv) > 4 else 0.005
    rclpy.init()
    node = Node("touch_table")
    cli = node.create_client(GoToTarget, "/go_to_target")
    cli.wait_for_service()

    def go(z_):
        req = GoToTarget.Request()
        req.target_pose.header.frame_id = "world"
        req.target_pose.pose.position.x, req.target_pose.pose.position.y, req.target_pose.pose.position.z = x, y, z_
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut)
        return fut.result().message

    print(f"-> ({x:.3f}, {y:.3f}, z={z:.3f}): {go(z)}")
    while True:
        a = input("ENTREE = descendre de %.0f mm | s = la pointe touche | q = annuler : " % (step * 1000)).strip().lower()
        if a == "q":
            print("annule")
            break
        if a == "s":
            print(f"\n>>> table a x={x:.3f}, y={y:.3f} : z = {z:.4f} m (pince verticale)")
            break
        z -= step
        print(f"-> z={z:.3f}: {go(z)}")


if __name__ == "__main__":
    main()
