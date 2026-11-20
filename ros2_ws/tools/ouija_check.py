#!/usr/bin/env python3
"""ouija_check.py - compares the Ouija perception with the simulator ground truth and
(optionally) checks the jaw alignment by taking camera frames.

    python3 /ros2_ws/tools/ouija_check.py            # planchette: perception vs GT
    python3 /ros2_ws/tools/ouija_check.py --jaws     # also saves /ros2_ws/jaws_x.png / jaws_y.png
"""
import argparse
import math
import sys
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from so101_interfaces.srv import GoToTarget, SetJointPositions

OBSERVE = [0.0, -100.0, 40.0, -80.0, 0.0]


def call(node, cli, req, timeout=60.0):
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
    return fut.result()


def pose_with_jaw(x, y, z, jaw):
    msg = PoseStamped()
    msg.header.frame_id = "world"
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = x, y, z
    xa = np.array([jaw[0], jaw[1], 0.0]); xa /= np.linalg.norm(xa)
    za = np.array([0.0, 0.0, -1.0]); ya = np.cross(za, xa)
    R = np.column_stack([xa, ya, za])
    w = math.sqrt(max(0.0, 1 + np.trace(R))) / 2
    if w > 1e-6:
        q = [(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w]
    else:
        qx, qy = math.sqrt(max(0.0, (1 + R[0, 0]) / 2)), math.sqrt(max(0.0, (1 + R[1, 1]) / 2))
        if R[0, 1] < 0:
            qy = -qy
        q = [qx, qy, 0.0, 0.0]
    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = map(float, q)
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jaws", action="store_true")
    args = ap.parse_args()
    rclpy.init()
    node = Node("ouija_check")
    latest = {}
    node.create_subscription(PoseStamped, "/ouija/planchette_position", lambda m: latest.update(det=m), 5)
    node.create_subscription(PoseStamped, "/debug/planchette_position_gt", lambda m: latest.update(gt=m), 5)
    node.create_subscription(PoseStamped, "/ouija/board_pose", lambda m: latest.update(board=m), 5)
    node.create_subscription(String, "/ouija/pointed", lambda m: latest.update(pointed=m), 5)
    node.create_subscription(Image, "/external_cam/image_raw", lambda m: latest.update(img=m), qos_profile_sensor_data)
    cli_j = node.create_client(SetJointPositions, "/set_joint_positions")
    cli_g = node.create_client(GoToTarget, "/go_to_target")
    cli_j.wait_for_service(); cli_g.wait_for_service()

    req = SetJointPositions.Request(); req.positions = OBSERVE + [60.0]
    print("observe:", call(node, cli_j, req).message)
    latest.pop("det", None)
    end = time.monotonic() + 4.0
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if "det" not in latest:
        print("planchette NOT detected"); 
    else:
        d, g = latest["det"].pose.position, latest["gt"].pose.position
        err = math.hypot(d.x - g.x, d.y - g.y)
        print(f"planchette: detected ({d.x:.4f}, {d.y:.4f})  gt ({g.x:.4f}, {g.y:.4f})  error {err * 1000:.1f} mm"
              f"  pointed='{latest.get('pointed', String()).data}'")
    b = latest["board"].pose.position
    print(f"board: ({b.x:.4f}, {b.y:.4f}, {b.z:.4f})  expected (0.2350, 0.0000, 0.0040)")

    if args.jaws:
        for name, jaw in (("x", (1.0, 0.0)), ("y", (0.0, 1.0))):
            req = GoToTarget.Request(); req.target_pose = pose_with_jaw(0.22, 0.0, 0.10, jaw)
            print(f"jaws along {name}:", call(node, cli_g, req).message)
            end = time.monotonic() + 1.0
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.1)
            img = CvBridge().imgmsg_to_cv2(latest["img"], "bgr8")
            cv2.imwrite(f"/ros2_ws/jaws_{name}.png", img)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
