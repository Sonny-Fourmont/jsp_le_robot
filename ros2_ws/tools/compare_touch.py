#!/usr/bin/env python3
"""compare_touch.py - compares where the camera sees the cube with where the arm says its tip is.
1. the script first records the cube seen by the camera (2 s, nothing in the way: keep the arm and
   your hands away from the cube) ;
2. then put the TIP of the gripper (held vertical, torque OFF) on the CENTRE OF THE TOP FACE of the
   cube and press ENTER: the tip is read from /end_effector_pose.
    python3 /ros2_ws/tools/compare_touch.py
Prints cube seen (before), tip, and the difference tip - camera."""
import statistics

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


def main():
    rclpy.init()
    node = Node("compare_touch")
    cam, tip = [], {}
    node.create_subscription(PoseStamped, "/object_position", lambda m: cam.append((m.pose.position.x, m.pose.position.y, m.pose.position.z)), 1)
    node.create_subscription(PoseStamped, "/end_effector_pose", lambda m: tip.update(p=m.pose.position), 1)
    t0 = node.get_clock().now()
    while (len(cam) < 1 or "p" not in tip) and (node.get_clock().now() - t0).nanoseconds < 20e9:
        rclpy.spin_once(node, timeout_sec=0.2)  # discovery can take a few seconds
    if not cam or "p" not in tip:
        raise SystemExit(f"missing: camera={'ok' if cam else 'no'} tip={'ok' if 'p' in tip else 'no'}")
    print("Je memorise la position du cube vue par la camera, ne touchez a rien (2 s)...")
    cam.clear()
    t1 = node.get_clock().now()
    while (node.get_clock().now() - t1).nanoseconds < 2e9:
        rclpy.spin_once(node, timeout_sec=0.1)
    if len(cam) < 5:
        raise SystemExit("cube non vu par la camera pendant la memorisation")
    cx, cy, cz = (statistics.median(v[i] for v in cam) for i in range(3))
    spread = max(max(v[0] for v in cam) - min(v[0] for v in cam), max(v[1] for v in cam) - min(v[1] for v in cam))
    print(f"cube vu par la camera : x={cx:.4f} y={cy:.4f} (dispersion {spread * 1000:.0f} mm, {len(cam)} mesures)")
    input("Posez la pointe sur le centre du dessus du cube, immobile, puis ENTREE... ")
    for _ in range(15):
        rclpy.spin_once(node, timeout_sec=0.1)
    t = tip["p"]
    print(f"pointe du bras        : x={t.x:.4f} y={t.y:.4f} z={t.z:.4f}   (dessus du cube attendu z={cz + 0.025:.4f})")
    print(f"ECART pointe - camera : dx={t.x - cx:+.4f} m  dy={t.y - cy:+.4f} m")


if __name__ == "__main__":
    main()
