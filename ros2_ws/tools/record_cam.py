#!/usr/bin/env python3
"""record_cam.py - saves one frame per second of /ouija/debug_image (or /external_cam) in a folder.
    python3 /ros2_ws/tools/record_cam.py /ros2_ws/rec 40    # folder, duration (s)"""
import sys
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
duration = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
rclpy.init()
node = Node("record_cam")
bridge = CvBridge()
state = {"last": 0.0, "n": 0}


def cb(msg):
    now = time.monotonic()
    if now - state["last"] >= 1.0:
        state["last"] = now
        cv2.imwrite(str(out / f"{state['n']:03d}.png"), bridge.imgmsg_to_cv2(msg, "bgr8"))
        state["n"] += 1


node.create_subscription(Image, "/ouija/debug_image", cb, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
end = time.monotonic() + duration
while time.monotonic() < end:
    rclpy.spin_once(node, timeout_sec=0.1)
print(f"{state['n']} frames in {out}")
rclpy.shutdown()
