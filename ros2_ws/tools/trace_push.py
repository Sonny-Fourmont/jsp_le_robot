#!/usr/bin/env python3
"""trace_push.py - records the TCP, the planchette (ground truth) and the Ouija state to a CSV.
    python3 /ros2_ws/tools/trace_push.py /ros2_ws/trace.csv 60"""
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String

path, duration = sys.argv[1], float(sys.argv[2])
rclpy.init()
node = Node("trace_push")
cur = {"state": "", "tcp": None, "puck": None}
node.create_subscription(String, "/ouija/state", lambda m: cur.update(state=m.data), 5)
node.create_subscription(PoseStamped, "/end_effector_pose", lambda m: cur.update(tcp=m.pose.position), 5)
node.create_subscription(PoseStamped, "/debug/planchette_position_gt", lambda m: cur.update(puck=m.pose.position), 5)
t0 = time.monotonic()
with open(path, "w") as f:
    f.write("t,state,tcp_x,tcp_y,tcp_z,puck_x,puck_y\n")
    last = 0.0
    while time.monotonic() - t0 < duration:
        rclpy.spin_once(node, timeout_sec=0.02)
        t = time.monotonic() - t0
        if t - last >= 0.1 and cur["tcp"] is not None and cur["puck"] is not None:
            last = t
            p, q = cur["tcp"], cur["puck"]
            f.write(f"{t:.2f},{cur['state']},{p.x:.4f},{p.y:.4f},{p.z:.4f},{q.x:.4f},{q.y:.4f}\n")
rclpy.shutdown()
