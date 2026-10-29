#!/usr/bin/env python3
"""make_calib_sheet.py - A4 sheet with ArUco markers to estimate the camera pose.

The subject says it: on the real robot nobody hands you the camera extrinsics. This
sheet, laid at a known place in the robot frame, lets `camera_calib` (so101_perception)
estimate the TF world -> camera with solvePnP on the 16 marker corners.

Sheet frame: origin at the centre of the sheet, x along the short side (21 cm, to the
right when the page is read in portrait), y along the long side (29.7 cm, towards the
top of the page), z out of the paper. The same frame is drawn on the sheet.

Outputs (next to this script):
    calib_sheet.png   texture for MuJoCo (also printable at 100 %)
    calib_sheet.pdf   A4 page, print at actual size (no "fit to page")
    calib_sheet.json  marker ids, size and centres in the sheet frame
    python3 make_calib_sheet.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

PX_PER_M = 6000  # 6 px / mm (152 dpi): fine for the markers and for the sim texture
SHEET_X, SHEET_Y = 0.210, 0.297  # m, A4 portrait: x short side, y long side
MARKER_SIZE = 0.07  # m, ArUco DICT_4X4_50
MARKER_INSET = 0.010  # m, from the paper edge
MARKER_IDS = {10: (+1, +1), 11: (-1, +1), 12: (-1, -1), 13: (+1, -1)}  # id -> corner (sign x, sign y)
# ids 0..3 are the Ouija board (ouija/make_board.py): keep them distinct.


def sheet_to_px(sx, sy):
    """Sheet frame (m) -> pixel (u right, v down). u = +x, v = -y: this is also how
    MuJoCo maps a 2D texture on the top face of a box (u along +x, v along -y), so the
    png is used as is, without rotation."""
    return int(round((sx + SHEET_X / 2) * PX_PER_M)), int(round((SHEET_Y / 2 - sy) * PX_PER_M))


def main():
    out = Path(__file__).parent
    W, H = int(round(SHEET_X * PX_PER_M)), int(round(SHEET_Y * PX_PER_M))
    img = np.full((H, W, 3), 255, np.uint8)
    black = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # marker_axes: directions of the marker image axes (u right, v down) in the sheet frame,
    # used by camera_calib to order the corners (top-left, top-right, bottom-right, bottom-left).
    layout = {"sheet_size": [SHEET_X, SHEET_Y], "marker_size": MARKER_SIZE, "dictionary": "DICT_4X4_50",
              "marker_axes": [[1, 0], [0, -1]], "markers": {}}
    aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    mpx = int(round(MARKER_SIZE * PX_PER_M))
    c_off = (SHEET_X / 2 - MARKER_INSET - MARKER_SIZE / 2, SHEET_Y / 2 - MARKER_INSET - MARKER_SIZE / 2)
    for mid, (sgx, sgy) in MARKER_IDS.items():
        cx, cy = sgx * c_off[0], sgy * c_off[1]
        marker = cv2.aruco.generateImageMarker(aruco, mid, mpx)
        u, v = sheet_to_px(cx, cy)
        img[v - mpx // 2:v - mpx // 2 + mpx, u - mpx // 2:u - mpx // 2 + mpx] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        layout["markers"][mid] = [round(cx, 4), round(cy, 4)]
        lv = v - mpx // 2 - 15 if sgy > 0 else v + mpx // 2 + 45  # label towards the paper edge
        cv2.putText(img, f"id {mid}", (u - mpx // 2, lv), font, 1.2, black, 2, cv2.LINE_AA)

    # Free band between the markers: |y| < 0.068. Frame at the centre: cross + axes
    # (x right, y up), titles above, scale bar below.
    ou, ov = sheet_to_px(0.0, 0.0)
    cv2.circle(img, (ou, ov), 12, black, 3)
    cv2.line(img, (ou - 40, ov), (ou + 40, ov), black, 3)
    cv2.line(img, (ou, ov - 40), (ou, ov + 40), black, 3)
    cv2.arrowedLine(img, (ou, ov), sheet_to_px(0.03, 0.0), black, 6, tipLength=0.2)
    cv2.arrowedLine(img, (ou, ov), sheet_to_px(0.0, 0.03), black, 6, tipLength=0.2)
    cv2.putText(img, "x", sheet_to_px(0.034, -0.004), font, 1.6, black, 3, cv2.LINE_AA)
    cv2.putText(img, "y", sheet_to_px(0.004, 0.036), font, 1.6, black, 3, cv2.LINE_AA)
    cv2.putText(img, "origine = centre de la feuille", sheet_to_px(-0.075, -0.016), font, 1.0, black, 2, cv2.LINE_AA)

    titles = ["SO-ARM101 - calibration camera (ArUco 4x4_50, 7 cm, ids 10-13)",
              "Imprimer a 100 % (taille reelle). sheet_pose = [x y z yaw] du centre"]
    for i, line in enumerate(titles):
        cv2.putText(img, line, sheet_to_px(-0.095, 0.062 - 0.012 * i), font, 0.8, black, 2, cv2.LINE_AA)

    # Scale bar (10 cm) to check the print size.
    a, b = sheet_to_px(-0.05, -0.045), sheet_to_px(0.05, -0.045)
    cv2.line(img, a, b, black, 4)
    for q in (a, b):
        cv2.line(img, (q[0], q[1] - 25), (q[0], q[1] + 25), black, 4)
    cv2.putText(img, "10 cm (verifier a l'impression)", sheet_to_px(-0.062, -0.058), font, 1.0, black, 2, cv2.LINE_AA)

    cv2.imwrite(str(out / "calib_sheet.png"), img)
    (out / "calib_sheet.json").write_text(json.dumps(layout, indent=1))
    try:
        from PIL import Image
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(
            str(out / "calib_sheet.pdf"), "PDF", resolution=PX_PER_M * 0.0254)
        pdf = "calib_sheet.pdf"
    except ImportError:
        pdf = "no pdf (PIL missing)"
    print(f"wrote {W}x{H} calib_sheet.png, {pdf}, {len(layout['markers'])} markers")


if __name__ == "__main__":
    main()
