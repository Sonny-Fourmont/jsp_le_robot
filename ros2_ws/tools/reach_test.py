#!/usr/bin/env python3
"""reach_test.py - asks the brain to reach a list of TCP positions and reports the errors.
    python3 /ros2_ws/tools/reach_test.py "0.10,0,0.02,0" "0.30,0,0.02,0" ...
Each argument: x,y,z[,tilt_deg]  (tilt: approach tilted forward from vertical)."""
import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from so101_interfaces.srv import GoToTarget


def main():
    rclpy.init()
    node = Node("reach_test")
    cli = node.create_client(GoToTarget, "/go_to_target")
    cli.wait_for_service()
    for arg in sys.argv[1:]:
        vals = [float(v) for v in arg.split(",")]
        x, y, z = vals[:3]
        tilt = math.radians(vals[3]) if len(vals) > 3 else 0.0
        req = GoToTarget.Request()
        req.target_pose.header.frame_id = "world"
        req.target_pose.pose.position.x, req.target_pose.pose.position.y, req.target_pose.pose.position.z = x, y, z
        a = math.pi - tilt  # rotation about the pan direction: z axis -> (sin a, 0, cos a) in the xz plane
        yaw = math.atan2(y, x)
        # tilt forward along the radial direction: rotate about the axis perpendicular to it
        ax = (-math.sin(yaw), math.cos(yaw), 0.0)
        s = math.sin(a / 2)
        req.target_pose.pose.orientation.x = ax[0] * s
        req.target_pose.pose.orientation.y = ax[1] * s
        req.target_pose.pose.orientation.z = ax[2] * s
        req.target_pose.pose.orientation.w = math.cos(a / 2)
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=30.0)
        r = fut.result()
        print(f"{arg:>22}: {'OK ' if r and r.success else 'KO '} {r.message if r else 'timeout'}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
