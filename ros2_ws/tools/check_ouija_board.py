#!/usr/bin/env python3
"""check_ouija_board.py - sanity check of the Ouija scene seen by the external camera.

Grabs one frame from /external_cam, detects the ArUco markers and compares their pixel
positions with the ground-truth projection (simulator extrinsics + layout file).
Optionally moves the arm to the observe pose first (--observe) and saves the frame (--save).

    source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash
    python3 /ros2_ws/tools/check_ouija_board.py --observe --save /ros2_ws/cam_debug.png
"""
import argparse
import json
import math
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from so101_sim import ASSETS_DIR, SO101Sim

OBSERVE_POSE_DEG = [0.0, -80.0, 80.0, 60.0, 0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observe", action="store_true", help="move the arm to the observe pose first")
    ap.add_argument("--save", default="", help="save the annotated frame to this path")
    ap.add_argument("--scene", default="scene_ouija.xml")
    ap.add_argument("--pose", default="", help="observe pose, 5 joint angles in deg (comma separated)")
    args = ap.parse_args()

    rclpy.init()
    node = Node("check_ouija_board")
    if args.observe:
        from so101_interfaces.srv import SetJointPositions

        cli = node.create_client(SetJointPositions, "/set_joint_positions")
        if not cli.wait_for_service(5.0):
            sys.exit("/set_joint_positions not available")
        req = SetJointPositions.Request()
        pose = [float(v) for v in args.pose.split(",")] if args.pose else OBSERVE_POSE_DEG
        req.positions = pose + [0.0]
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=20.0)
        print("observe pose:", fut.result().message if fut.done() else "timeout")

    frame = {}
    node.create_subscription(Image, "/external_cam/image_raw", lambda m: frame.setdefault("img", m), 1)
    node.create_subscription(CameraInfo, "/external_cam/camera_info", lambda m: frame.setdefault("info", m), 1)
    end = node.get_clock().now().nanoseconds + 5_000_000_000
    while ("img" not in frame or "info" not in frame) and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if "img" not in frame:
        sys.exit("no image on /external_cam/image_raw")
    bgr = CvBridge().imgmsg_to_cv2(frame["img"], "bgr8")
    K = np.array(frame["info"].k).reshape(3, 3)

    # Ground truth projection, computed from a private copy of the simulator.
    sim = SO101Sim(scene_path=ASSETS_DIR / args.scene)
    sim.connect()
    T_wc = sim.get_camera_extrinsics() @ np.diag([1.0, -1.0, -1.0, 1.0])
    T_cw = np.linalg.inv(T_wc)
    layout = json.loads((ASSETS_DIR / "ouija" / "ouija_layout.json").read_text())
    cx, cy, cz = layout["board_center_world"]

    def project(p_world):
        pc = T_cw @ np.array([*p_world, 1.0])
        uv = K @ pc[:3]
        return uv[0] / uv[2], uv[1] / uv[2]

    gt_markers = {int(k): project((cx + bx, cy + by, 0.004)) for k, (bx, by) in layout["markers"].items()}

    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = det.detectMarkers(bgr)
    found = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            found[int(i)] = c[0].mean(axis=0)
    ok = True
    for mid in sorted(gt_markers):
        gu, gv = gt_markers[mid]
        if mid in found:
            du, dv = found[mid]
            err = math.hypot(du - gu, dv - gv)
            ok &= err < 3.0
            print(f"marker {mid}: detected ({du:6.1f},{dv:6.1f})  gt ({gu:6.1f},{gv:6.1f})  err {err:4.1f} px")
        else:
            ok = False
            print(f"marker {mid}: NOT detected (gt at ({gu:6.1f},{gv:6.1f}))")
    extra = sorted(set(found) - set(gt_markers))
    if extra:
        print("spurious ids (ignored by the perception node):", extra)
    for lab in ("A", "M", "N", "Z", "0", "9", "OUI", "FIN", "NON"):
        bx, by = layout["targets"][lab]
        print(f"  {lab:>3}: gt px {tuple(round(v, 1) for v in project((cx + bx, cy + by, 0.004)))}")
    if args.save:
        cv2.aruco.drawDetectedMarkers(bgr, corners, ids)
        for lab, (bx, by) in layout["targets"].items():
            u, v = project((cx + bx, cy + by, 0.004))
            cv2.circle(bgr, (int(u), int(v)), 3, (0, 0, 255), -1)
        cv2.imwrite(args.save, bgr)
        print("saved", args.save)
    print("BOARD OK" if ok else "BOARD MISMATCH")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
