#!/usr/bin/env python3
"""monitor_positions.py - shows the raw position (0..4095) of the 6 motors, live, read-only.
The driver must be stopped. Ctrl+C to quit. Move one joint at a time and watch its column:
a full range of motion is ~2200-2500 ticks (4096 = 360 deg).
    docker compose exec control bash -lc "source /opt/ros/jazzy/setup.bash && python3 /ros2_ws/tools/monitor_positions.py"
"""
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    bus = FeetechMotorsBus(port="/dev/ttyACM0", motors={
        n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(NAMES)})
    bus.connect()
    bus.disable_torque()
    lo = hi = None
    print("   ".join(f"{n[:12]:>12s}" for n in NAMES), "   (ligne 2: min..max vus)")
    try:
        while True:
            p = bus.sync_read("Present_Position", NAMES, normalize=False)
            v = [p[n] for n in NAMES]
            lo = v[:] if lo is None else [min(a, b) for a, b in zip(lo, v)]
            hi = v[:] if hi is None else [max(a, b) for a, b in zip(hi, v)]
            print("\r" + "   ".join(f"{x:12d}" for x in v) + "   | écart vu: " + " ".join(f"{h - l:5d}" for l, h in zip(lo, hi)), end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
