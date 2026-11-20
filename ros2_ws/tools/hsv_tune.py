#!/usr/bin/env python3
"""hsv_tune.py - tune the HSV thresholds of the perception on the live camera image.

Shows the camera image with the detected blob, and the mask of the selected target, with
sliders. Keys: 1 = cube (red), 2 = box, 3 = ring (Ouija planchette), s = save, q = quit.
Saving writes ros2_ws/config/perception_real.yaml, loaded on the real robot
(PERCEPTION_PARAMS in docker/.env); then: docker compose restart perception

    ./run.sh hsv                                          # real camera (camera container, X11)
    python3 hsv_tune.py --snapshot /ros2_ws/rec/hsv.png   # no GUI: one frame + the 3 masks with the current yaml
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import rclpy.executors
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

YAML = Path("/ros2_ws/config/perception_real.yaml")
# Sliders per target. Cube: the red hue wraps around 0, so one range [h_low..179] + [0..h_high]
# with common S/V minimums (that is how perception_node's two ranges are built).
SLIDERS = {
    "cube": [("h_low", 179), ("h_high", 179), ("s_min", 255), ("v_min", 255)],
    "box": [("h_lo", 179), ("h_hi", 179), ("s_lo", 255), ("s_hi", 255), ("v_lo", 255), ("v_hi", 255)],
    "ring": [("h_lo", 179), ("h_hi", 179), ("s_lo", 255), ("s_hi", 255), ("v_lo", 255), ("v_hi", 255)],
}
DEFAULTS = {
    "cube": {"h_low": 160, "h_high": 8, "s_min": 90, "v_min": 60},
    "box": {"h_lo": 5, "h_hi": 22, "s_lo": 80, "s_hi": 255, "v_lo": 40, "v_hi": 160},
    "ring": {"h_lo": 100, "h_hi": 130, "s_lo": 120, "s_hi": 255, "v_lo": 80, "v_hi": 255},
}
KEYS = {ord("1"): "cube", ord("2"): "box", ord("3"): "ring"}


def load_values(path):
    """Current yaml -> slider values (defaults where missing)."""
    vals = {k: dict(v) for k, v in DEFAULTS.items()}
    if not path.exists():
        return vals
    y = yaml.safe_load(path.read_text()) or {}
    p = y.get("so101_perception", {}).get("ros__parameters", {})
    o = y.get("ouija_perception", {}).get("ros__parameters", {})
    if "object_hsv_low1" in p:
        lo1, hi2 = p["object_hsv_low1"], p.get("object_hsv_high2", [8, 255, 255])
        vals["cube"] = {"h_low": lo1[0], "h_high": hi2[0], "s_min": lo1[1], "v_min": lo1[2]}
    if "box_hsv_low" in p:
        lo, hi = p["box_hsv_low"], p["box_hsv_high"]
        vals["box"] = dict(zip(["h_lo", "s_lo", "v_lo"], lo)) | dict(zip(["h_hi", "s_hi", "v_hi"], hi))
    if "planchette_hsv_low" in o:
        lo, hi = o["planchette_hsv_low"], o["planchette_hsv_high"]
        vals["ring"] = dict(zip(["h_lo", "s_lo", "v_lo"], lo)) | dict(zip(["h_hi", "s_hi", "v_hi"], hi))
    return vals


def save_values(path, vals):
    c, b, r = vals["cube"], vals["box"], vals["ring"]
    old = {}
    if path.exists():
        old = (yaml.safe_load(path.read_text()) or {}).get("so101_perception", {}).get("ros__parameters", {})
    doc = {
        "so101_perception": {"ros__parameters": {
            "object_hsv_low1": [c["h_low"], c["s_min"], c["v_min"]], "object_hsv_high1": [179, 255, 255],
            "object_hsv_low2": [0, c["s_min"], c["v_min"]], "object_hsv_high2": [c["h_high"], 255, 255],
            "box_hsv_low": [b["h_lo"], b["s_lo"], b["v_lo"]], "box_hsv_high": [b["h_hi"], b["s_hi"], b["v_hi"]],
            "min_blob_area_px": int(old.get("min_blob_area_px", 40)),
            "object_plane_z": float(old.get("object_plane_z", 0.025)),
            "box_plane_z": float(old.get("box_plane_z", 0.025)),
        }},
        "ouija_perception": {"ros__parameters": {
            "planchette_hsv_low": [r["h_lo"], r["s_lo"], r["v_lo"]],
            "planchette_hsv_high": [r["h_hi"], r["s_hi"], r["v_hi"]],
        }},
    }
    header = ("# HSV thresholds for the REAL camera (OpenCV: H 0-179, S 0-255, V 0-255), written by\n"
              f"# tools/hsv_tune.py on {time.strftime('%Y-%m-%d %H:%M')}. Loaded when PERCEPTION_PARAMS points\n"
              "# here (docker/.env). After a change: docker compose restart perception\n")
    path.write_text(header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))


def masks(bgr, vals):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    c, out = vals["cube"], {}
    out["cube"] = (cv2.inRange(hsv, (c["h_low"], c["s_min"], c["v_min"]), (179, 255, 255))
                   | cv2.inRange(hsv, (0, c["s_min"], c["v_min"]), (c["h_high"], 255, 255)))
    for k in ("box", "ring"):
        v = vals[k]
        out[k] = cv2.inRange(hsv, (v["h_lo"], v["s_lo"], v["v_lo"]), (v["h_hi"], v["s_hi"], v["v_hi"]))
    kernel = np.ones((5, 5), np.uint8)
    return {k: cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel) for k, m in out.items()}


def largest_blob(mask, min_area=40):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    a = cv2.contourArea(c)
    if a < min_area:
        return None
    m = cv2.moments(c)
    return c, a, (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))


COLORS = {"cube": (0, 0, 255), "box": (0, 165, 255), "ring": (255, 128, 0)}


def annotate(bgr, ms):
    img = bgr.copy()
    for k, m in ms.items():
        b = largest_blob(m)
        if b is None:
            continue
        c, a, (u, v) = b
        cv2.drawContours(img, [c], -1, COLORS[k], 2)
        cv2.circle(img, (u, v), 4, COLORS[k], -1)
        cv2.putText(img, f"{k} ({u},{v}) {a:.0f}px", (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[k], 2)
    return img


class Cam(Node):
    def __init__(self, topic):
        super().__init__("hsv_tune")
        self.frame, self.lock = None, threading.Lock()
        self.create_subscription(Image, topic, self._cb, qos_profile_sensor_data)

    def _cb(self, msg):
        buf = np.frombuffer(msg.data, np.uint8)
        if msg.encoding == "mono8":
            img = cv2.cvtColor(buf.reshape(msg.height, msg.step)[:, : msg.width], cv2.COLOR_GRAY2BGR)
        else:
            img = buf.reshape(msg.height, msg.step // 3, 3)[:, : msg.width]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else img.copy()
        with self.lock:
            self.frame = img

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", default="/external_cam/image_raw")
    ap.add_argument("--yaml", default=str(YAML))
    ap.add_argument("--snapshot", help="no GUI: write one annotated frame (image + 3 masks) to this png and exit")
    args = ap.parse_args()
    path = Path(args.yaml)
    vals = load_values(path)

    rclpy.init()
    cam = Cam(args.topic)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(cam)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    def shutdown():
        executor.shutdown()
        spin.join(timeout=2.0)
        cam.destroy_node()
        rclpy.shutdown()
    t0 = time.time()
    while cam.latest() is None and time.time() - t0 < 15:
        time.sleep(0.1)
    frame = cam.latest()
    if frame is None:
        print(f"no image on {args.topic} after 15 s (camera running? QoS?)")
        shutdown()
        sys.exit(1)

    if args.snapshot:
        ms = masks(frame, vals)
        top = np.hstack([annotate(frame, ms), cv2.cvtColor(ms["cube"], cv2.COLOR_GRAY2BGR)])
        bot = np.hstack([cv2.cvtColor(ms["box"], cv2.COLOR_GRAY2BGR), cv2.cvtColor(ms["ring"], cv2.COLOR_GRAY2BGR)])
        grid = np.vstack([top, bot])
        for i, label in enumerate(["image + blobs", "masque cube", "masque boite", "masque anneau"]):
            x, y = (i % 2) * frame.shape[1] + 10, (i // 2) * frame.shape[0] + 25
            cv2.putText(grid, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(args.snapshot, grid)
        for k, m in ms.items():
            b = largest_blob(m)
            print(f"{k}: " + (f"blob at {b[2]}, {b[1]:.0f} px" if b else "not detected"))
        print(f"wrote {args.snapshot}")
        shutdown()
        return

    win = "hsv_tune"
    target = "cube"

    def build(target):
        cv2.destroyAllWindows()
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 2 * frame.shape[1], frame.shape[0] + 40)
        for name, vmax in SLIDERS[target]:
            cv2.createTrackbar(name, win, int(vals[target][name]), vmax, lambda _v: None)

    build(target)
    print("keys: 1 cube, 2 box, 3 ring, s save to yaml, q quit")
    while True:
        for name, _ in SLIDERS[target]:
            vals[target][name] = cv2.getTrackbarPos(name, win)
        frame = cam.latest()
        ms = masks(frame, vals)
        view = np.hstack([annotate(frame, ms), cv2.cvtColor(ms[target], cv2.COLOR_GRAY2BGR)])
        txt = f"[{target}] " + "  ".join(f"{k}={v}" for k, v in vals[target].items()) + "   1/2/3 cible, s sauver, q quitter"
        cv2.putText(view, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            save_values(path, vals)
            print(f"saved {path}; apply with: docker compose restart perception")
        if key in KEYS and KEYS[key] != target:
            target = KEYS[key]
            build(target)
    cv2.destroyAllWindows()
    shutdown()


if __name__ == "__main__":
    main()
