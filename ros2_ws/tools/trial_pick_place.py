#!/usr/bin/env python3
"""trial_pick_place.py - N randomized pick & place trials in simulation, checked with ground truth.

    docker compose exec control bash -lc "source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash && python3 /ros2_ws/tools/trial_pick_place.py 10"
"""
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_srvs.srv import Trigger


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rclpy.init()
    node = Node("trial_pick_place")
    gt = {}

    def store(key):
        return lambda m: gt.__setitem__(key, np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z]))

    node.create_subscription(PoseStamped, "/debug/object_position_gt", store("o"), 5)
    node.create_subscription(PoseStamped, "/debug/drop_box_position_gt", store("b"), 5)
    randomize = node.create_client(Trigger, "/sim/randomize_object")
    pick = node.create_client(Trigger, "/pick_and_place")
    randomize.wait_for_service(10)
    pick.wait_for_service(10)

    def call(client):
        fut = client.call_async(Trigger.Request())
        while not fut.done():
            rclpy.spin_once(node, timeout_sec=0.05)
        return fut.result()

    def spin(seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    ok = 0
    fails = []
    for i in range(n_trials):
        call(randomize)
        spin(1.5)
        start = gt["o"].copy()
        t0 = time.time()
        res = call(pick)
        dt = time.time() - t0
        spin(1.0)
        o, b = gt["o"], gt["b"]
        in_box = np.linalg.norm((o - b)[:2]) < 0.05 and o[2] < 0.06
        ok += in_box
        r, a = np.hypot(*start[:2]), np.degrees(np.arctan2(start[1], start[0]))
        verdict = "IN BOX" if in_box else "NOT in box"
        print(f"trial {i + 1}: start r={r:.2f} a={a:.0f}deg -> {res.success} in {dt:.0f}s, {verdict}")
        if not in_box or not res.success:
            fails.append((np.round(start[:2], 2), res.success, res.message[:90], np.round(o, 3)))
    print(f"success {ok}/{n_trials}")
    for f in fails:
        print("FAIL", f)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
