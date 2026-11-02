#!/usr/bin/env python3
"""make_board.py - generates the Ouija board texture and the layout file.

Board frame: origin at the board centre, x towards the far edge (robot +x when the
board is placed in front of the robot), y to the left. The letter arcs are centred on
the robot base, which sits at (-BOARD_CX, 0) in the board frame, so every letter of an
arc is at a constant radius from the robot.

Outputs (next to this script): ouija_board.png and ouija_layout.json
    python3 make_board.py
Printable version on 4 A4 pages: make_board_print.py
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


def world_to_px(x, y, px_per_m=PX_PER_M):
    """World (x forward, y left) -> pixel of the drawing canvas (u right, v down).
    The canvas is drawn as the overhead camera sees the board: up = +x (far edge),
    right = -y. It is rotated afterwards to match MuJoCo's box texture mapping."""
    bx, by = x - BOARD_CX, y  # board frame
    u = (BOARD_Y / 2 - by) * px_per_m
    v = (BOARD_X / 2 - bx) * px_per_m
    return int(round(u)), int(round(v))


def draw_board(px_per_m=PX_PER_M, background=(120, 170, 215)):
    """Draws the board canvas (BOARD_Y wide, BOARD_X tall, camera-aligned) and returns
    (image, layout). background: BGR, wood-ish for the sim texture, white for printing."""
    k = px_per_m / 3000.0  # line widths were tuned at 3000 px/m
    W, H = int(round(BOARD_Y * px_per_m)), int(round(BOARD_X * px_per_m))
    img = np.full((H, W, 3), background, np.uint8)
    m = int(30 * k)
    cv2.rectangle(img, (m, m), (W - m, H - m), (20, 20, 20), int(6 * k))

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
            u, v = world_to_px(x, y, px_per_m)
            scale = (2.2 if len(lab) == 1 else 1.3) * k
            (tw, th), _ = cv2.getTextSize(lab, font, scale, int(5 * k))
            cv2.putText(img, lab, (u - tw // 2, v + th // 2), font, scale, (15, 15, 15), int(5 * k), cv2.LINE_AA)
    # decorative star in the middle (between arcs)
    cu, cv_ = world_to_px(0.236, 0.0, px_per_m)
    cv2.circle(img, (cu, cv_), int(40 * k), (15, 15, 15), int(3 * k))

    aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    mpx = int(round(MARKER_SIZE * px_per_m))
    corners_board = {  # marker centre in board frame (x, y)
        0: (BOARD_X / 2 - MARKER_INSET - MARKER_SIZE / 2, BOARD_Y / 2 - MARKER_INSET - MARKER_SIZE / 2),
        1: (BOARD_X / 2 - MARKER_INSET - MARKER_SIZE / 2, -(BOARD_Y / 2 - MARKER_INSET - MARKER_SIZE / 2)),
        2: (-(BOARD_X / 2 - MARKER_INSET - MARKER_SIZE / 2), -(BOARD_Y / 2 - MARKER_INSET - MARKER_SIZE / 2)),
        3: (-(BOARD_X / 2 - MARKER_INSET - MARKER_SIZE / 2), BOARD_Y / 2 - MARKER_INSET - MARKER_SIZE / 2),
    }
    for mid, (bx, by) in corners_board.items():
        marker = cv2.aruco.generateImageMarker(aruco, mid, mpx)
        u, v = world_to_px(bx + BOARD_CX, by, px_per_m)
        # marker image axes: +u right, +v down in the texture; in board frame u = -y, v = -x.
        img[v - mpx // 2:v - mpx // 2 + mpx, u - mpx // 2:u - mpx // 2 + mpx] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        layout["markers"][mid] = [round(bx, 4), round(by, 4)]
    return img, layout


def main():
    out = Path(__file__).parent
    img, layout = draw_board()
    # MuJoCo maps a 2D texture on the top face of a box with texture u along the geom
    # +x axis and texture v along the geom -y axis (checked with the ArUco markers).
    # Our canvas has u = -y and v = -x, i.e. the texture is the canvas rotated 90 deg
    # clockwise. The png is then BOARD_X wide and BOARD_Y tall: no stretching.
    tex = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(str(out / "ouija_board.png"), tex)
    W, H = img.shape[1], img.shape[0]
    cv2.imwrite(str(out / "ouija_board_preview.png"), cv2.resize(img, (W // 3, H // 3)))
    (out / "ouija_layout.json").write_text(json.dumps(layout, indent=1))
    print(f"wrote {tex.shape[1]}x{tex.shape[0]} texture, {len(layout['targets'])} targets, {len(layout['markers'])} markers")


if __name__ == "__main__":
    main()
