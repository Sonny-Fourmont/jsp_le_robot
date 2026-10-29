#!/usr/bin/env python3
"""make_board.py - generates the Ouija board texture and the layout file.

Board frame: origin at the board centre, x towards the far edge (robot +x when the
board is placed in front of the robot), y to the left. The letter arcs are centred on
the robot base, which sits at (-BOARD_CX, 0) in the board frame, so every letter of an
arc is at a constant radius from the robot.

Outputs (next to this script): ouija_board.png and ouija_layout.json
    python3 make_board.py
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

PX_PER_M = 3000  # 3 px / mm
BOARD_X, BOARD_Y = 0.30, 0.50  # m (depth along x, width along y)
BOARD_CX = 0.235  # board centre is at world x = 0.235 (robot base at world origin)
MARKER_SIZE = 0.04  # m, ArUco DICT_4X4_50 ids 0..3 at the corners
MARKER_INSET = 0.012
# Planchette = ring (see scene_ouija.xml): outer radius, hole radius, height (m)
PLANCHETTE_RADIUS, PLANCHETTE_HOLE_RADIUS, PLANCHETTE_HEIGHT = 0.021, 0.011, 0.015

# Reach of the TCP pointing down (measured with tools/reach_test.py): 0.13 m .. 0.34 m from
# the base. The targets stay within 0.18 .. 0.30 m.
ARCS = [  # (labels, radius from robot base, angle start, angle end) in degrees, +angle = +y (left)
    (list("ABCDEFGHIJKLM"), 0.295, 45, -45),
    (list("NOPQRSTUVWXYZ"), 0.255, 45, -45),
    (list("0123456789"), 0.215, 40, -40),
    (["OUI", "FIN", "NON"], 0.180, 40, -40),
]


def world_to_px(x, y):
    """World (x forward, y left) -> pixel of the drawing canvas (u right, v down).
    The canvas is drawn as the overhead camera sees the board: up = +x (far edge),
    right = -y. It is rotated afterwards to match MuJoCo's box texture mapping."""
    bx, by = x - BOARD_CX, y  # board frame
    u = (BOARD_Y / 2 - by) * PX_PER_M
    v = (BOARD_X / 2 - bx) * PX_PER_M
    return int(round(u)), int(round(v))


def main():
    out = Path(__file__).parent
    W, H = int(BOARD_Y * PX_PER_M), int(BOARD_X * PX_PER_M)
    img = np.full((H, W, 3), (120, 170, 215), np.uint8)  # wood-ish (BGR)
    cv2.rectangle(img, (30, 30), (W - 30, H - 30), (20, 20, 20), 6)

    layout = {"board_size": [BOARD_X, BOARD_Y], "board_center_world": [BOARD_CX, 0.0, 0.0],
              "marker_size": MARKER_SIZE, "planchette_radius": PLANCHETTE_RADIUS,
              "planchette_hole_radius": PLANCHETTE_HOLE_RADIUS, "planchette_height": PLANCHETTE_HEIGHT,
              "start": "FIN", "marker_axes": [[0, -1], [-1, 0]],  # marker image u = -y, v = -x (camera_calib)
              "markers": {}, "targets": {}}
    font = cv2.FONT_HERSHEY_TRIPLEX
    for labels, r, a0, a1 in ARCS:
        n = len(labels)
        for i, lab in enumerate(labels):
            a = math.radians(a0 + (a1 - a0) * i / (n - 1))
            x, y = r * math.cos(a), r * math.sin(a)
            layout["targets"][lab] = [round(x - BOARD_CX, 4), round(y, 4)]  # board frame
            u, v = world_to_px(x, y)
            scale = 2.2 if len(lab) == 1 else 1.3
            (tw, th), _ = cv2.getTextSize(lab, font, scale, 5)
            cv2.putText(img, lab, (u - tw // 2, v + th // 2), font, scale, (15, 15, 15), 5, cv2.LINE_AA)
    # decorative star in the middle (between arcs)
    cu, cv_ = world_to_px(0.236, 0.0)
    cv2.circle(img, (cu, cv_), 40, (15, 15, 15), 3)

    aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    mpx = int(MARKER_SIZE * PX_PER_M)
    corners_board = {  # marker centre in board frame (x, y)
        0: (BOARD_X / 2 - MARKER_INSET - MARKER_SIZE / 2, BOARD_Y / 2 - MARKER_INSET - MARKER_SIZE / 2),
        1: (BOARD_X / 2 - MARKER_INSET - MARKER_SIZE / 2, -(BOARD_Y / 2 - MARKER_INSET - MARKER_SIZE / 2)),
        2: (-(BOARD_X / 2 - MARKER_INSET - MARKER_SIZE / 2), -(BOARD_Y / 2 - MARKER_INSET - MARKER_SIZE / 2)),
        3: (-(BOARD_X / 2 - MARKER_INSET - MARKER_SIZE / 2), BOARD_Y / 2 - MARKER_INSET - MARKER_SIZE / 2),
    }
    for mid, (bx, by) in corners_board.items():
        marker = cv2.aruco.generateImageMarker(aruco, mid, mpx)
        u, v = world_to_px(bx + BOARD_CX, by)
        # marker image axes: +u right, +v down in the texture; in board frame u = -y, v = -x.
        img[v - mpx // 2:v - mpx // 2 + mpx, u - mpx // 2:u - mpx // 2 + mpx] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        layout["markers"][mid] = [round(bx, 4), round(by, 4)]
    # MuJoCo maps a 2D texture on the top face of a box with texture u along the geom
    # +x axis and texture v along the geom -y axis (checked with the ArUco markers).
    # Our canvas has u = -y and v = -x, i.e. the texture is the canvas rotated 90 deg
    # clockwise. The png is then BOARD_X wide and BOARD_Y tall: no stretching.
    tex = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(str(out / "ouija_board.png"), tex)
    cv2.imwrite(str(out / "ouija_board_preview.png"), cv2.resize(img, (W // 3, H // 3)))
    (out / "ouija_layout.json").write_text(json.dumps(layout, indent=1))
    print(f"wrote {tex.shape[1]}x{tex.shape[0]} texture, {len(layout['targets'])} targets, {len(layout['markers'])} markers")


if __name__ == "__main__":
    main()
