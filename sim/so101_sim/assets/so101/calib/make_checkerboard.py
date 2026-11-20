#!/usr/bin/env python3
"""make_checkerboard.py - A4 (landscape) checkerboard for the camera intrinsics.

8 x 6 INNER corners = 9 x 7 squares of 25 mm (225 x 175 mm), i.e. the arguments of
    ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025 ...
Print at 100 % (no "fit to page"), check the 10 cm bar, glue on something flat and rigid.

Outputs (next to this script): checkerboard_8x6_25mm.png / .pdf
    python3 make_checkerboard.py
"""
from pathlib import Path

import cv2
import numpy as np

PX_PER_M = 6000  # 6 px / mm, like make_calib_sheet.py
PAGE_X, PAGE_Y = 0.297, 0.210  # m, A4 landscape
SQUARE = 0.025  # m
COLS, ROWS = 9, 7  # squares -> 8 x 6 inner corners


def main():
    out = Path(__file__).parent
    W, H = int(round(PAGE_X * PX_PER_M)), int(round(PAGE_Y * PX_PER_M))
    img = np.full((H, W, 3), 255, np.uint8)
    sq = int(round(SQUARE * PX_PER_M))
    bw, bh = COLS * sq, ROWS * sq
    x0, y0 = (W - bw) // 2, (H - bh) // 2 - 60  # a little above centre, text below
    for r in range(ROWS):
        for c in range(COLS):
            if (r + c) % 2 == 0:
                img[y0 + r * sq:y0 + (r + 1) * sq, x0 + c * sq:x0 + (c + 1) * sq] = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    ty = y0 + bh + 70
    cv2.putText(img, "Damier 8x6 coins internes, cases de 25 mm - imprimer a 100 % (taille reelle)",
                (x0, ty), font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "cameracalibrator --size 8x6 --square 0.025", (x0, ty + 45), font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    # 10 cm bar to check the print scale
    a = (x0 + bw - int(0.10 * PX_PER_M), ty + 30)
    b = (x0 + bw, ty + 30)
    cv2.line(img, a, b, (0, 0, 0), 4)
    for q in (a, b):
        cv2.line(img, (q[0], q[1] - 20), (q[0], q[1] + 20), (0, 0, 0), 4)
    cv2.putText(img, "10 cm", (a[0] + 200, ty + 75), font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(str(out / "checkerboard_8x6_25mm.png"), img)
    try:
        from PIL import Image
        Image.fromarray(img[..., 0]).save(str(out / "checkerboard_8x6_25mm.pdf"), "PDF", resolution=PX_PER_M * 0.0254)
        pdf = "checkerboard_8x6_25mm.pdf"
    except ImportError:
        pdf = "no pdf (PIL missing)"
    print(f"wrote {W}x{H} checkerboard_8x6_25mm.png, {pdf}: {COLS}x{ROWS} squares of {SQUARE * 1000:.0f} mm")


if __name__ == "__main__":
    main()
