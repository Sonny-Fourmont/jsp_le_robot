#!/usr/bin/env python3
"""make_board_print.py - the Ouija board (30 x 50 cm) on 4 A4 pages, to print at 100 %.

The board is split into 4 quarters of 15 x 25 cm, one per A4 portrait page, with the
exact geometry of make_board.py (the letters sit at radii matched to the arm's reach,
so the board must not be scaled). Cut each page along the black frame of its quarter
and assemble the 4 quarters edge to edge (page labels: avant/arrière, gauche/droite,
seen from the robot). White background to save ink.

Output: ouija_board_A4.pdf (4 pages) next to this script.
    python3 make_board_print.py
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from make_board import BOARD_X, BOARD_Y, draw_board

PX_PER_M = 6000  # 152 dpi
A4 = (0.210, 0.297)  # m, portrait


def main():
    out = Path(__file__).parent
    img, _ = draw_board(PX_PER_M, background=(255, 255, 255))
    # Same orientation as the texture: BOARD_X (30 cm) wide, BOARD_Y (50 cm) tall. On this
    # sheet "up" is the robot's left (+y) and "right" is the robot's front (+x).
    sheet = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    H, W = sheet.shape[:2]
    tw, th = W // 2, H // 2  # tile 15 x 25 cm
    pw, ph = int(round(A4[0] * PX_PER_M)), int(round(A4[1] * PX_PER_M))
    ox, oy = (pw - tw) // 2, (ph - th) // 2
    names = {(0, 0): "arriere-gauche", (0, 1): "avant-gauche", (1, 0): "arriere-droite", (1, 1): "avant-droite"}
    pages = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in range(2):
        for c in range(2):
            page = np.full((ph, pw, 3), 255, np.uint8)
            page[oy:oy + th, ox:ox + tw] = sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            cv2.rectangle(page, (ox, oy), (ox + tw - 1, oy + th - 1), (0, 0, 0), 2)  # cut line
            for (x, y) in ((ox, oy), (ox + tw, oy), (ox, oy + th), (ox + tw, oy + th)):  # crop marks
                cv2.line(page, (x - 80, y), (x + 80, y), (0, 0, 0), 2)
                cv2.line(page, (x, y - 80), (x, y + 80), (0, 0, 0), 2)
            n = len(pages) + 1
            cv2.putText(page, f"Ouija 30 x 50 cm - page {n}/4 : quart {names[(r, c)]} (vu du robot)",
                        (ox + 100, oy - 95), font, 0.85, (0, 0, 0), 2, cv2.LINE_AA)  # inside printer margins
            cv2.putText(page, "Imprimer a 100 % (taille reelle), couper sur le cadre noir, assembler bord a bord",
                        (ox + 100, oy - 55), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(page, "haut de la page = gauche du robot (+y), droite = avant du robot (+x)",
                        (ox + 100, oy + th + 60), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            pages.append(Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB)))
    pages[0].save(str(out / "ouija_board_A4.pdf"), "PDF", resolution=PX_PER_M * 0.0254, save_all=True,
                  append_images=pages[1:])
    cv2.imwrite(str(out / "ouija_board_print_preview.png"), cv2.resize(sheet, (W // 6, H // 6)))
    print(f"wrote ouija_board_A4.pdf: 4 pages, tiles {tw / PX_PER_M * 100:.0f} x {th / PX_PER_M * 100:.0f} cm")


if __name__ == "__main__":
    main()
