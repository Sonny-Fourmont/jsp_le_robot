#!/usr/bin/env python3
"""read_motor_calibration.py - reads the calibration stored in the arm's motors (read-only: no torque,
no write), compares it with the calibration file used by the driver and saves it as a new file.

The driver must not hold the port (only one program can open it): run this when the driver has
crashed on a calibration mismatch, or after stopping it. From docker/:
    docker compose exec control bash -lc \
        "source /opt/ros/jazzy/setup.bash && python3 /ros2_ws/tools/read_motor_calibration.py"
Output: ros2_ws/calibration/so101_follower_from_motors_<date>.json (to copy in calibration/ and
to set as CALIBRATION_FILE in docker/.env).
"""
import datetime
import json
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

PORT = "/dev/ttyACM0"
CURRENT_FILE = Path("/calibration/arm.json")
OUT_DIR = Path("/ros2_ws/calibration")
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    motors = {n: Motor(i + 1, "sts3215", MotorNormMode.RADIANS if n != "gripper" else MotorNormMode.RANGE_0_100)
              for i, n in enumerate(NAMES)}
    bus = FeetechMotorsBus(port=PORT, motors=motors)
    bus.connect()
    try:
        read = bus.read_calibration()
    finally:
        bus.disconnect(disable_torque=False)

    current = json.loads(CURRENT_FILE.read_text()) if CURRENT_FILE.is_file() else {}
    print(f"{'motor':15s} {'homing_offset (file -> motor)':32s} {'range_min':18s} {'range_max':18s}")
    for n in NAMES:
        c, m = current.get(n, {}), read[n]
        print(f"{n:15s} {str(c.get('homing_offset')):>10s} -> {m.homing_offset:<14d} "
              f"{str(c.get('range_min')):>6s} -> {m.range_min:<8d} {str(c.get('range_max')):>6s} -> {m.range_max:<8d}")

    data = {n: {"id": read[n].id, "drive_mode": read[n].drive_mode, "homing_offset": read[n].homing_offset,
                "range_min": read[n].range_min, "range_max": read[n].range_max} for n in NAMES}
    out = OUT_DIR / f"so101_follower_from_motors_{datetime.date.today().isoformat()}.json"
    out.write_text(json.dumps(data, indent=4) + "\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
