#!/usr/bin/env python3
"""write_motor_calibration.py - writes the calibration file used by the driver (/calibration/arm.json,
= CALIBRATION_FILE of docker/.env) INTO the motors (homing offsets and position limits).

Use it when the motors have lost their calibration (factory ranges 0..4095, other offsets) while the
calibration file is still the right one: no need to move the arm, no new calibration.
Torque is disabled (the arm goes limp); the driver must not hold the port.

    docker compose exec control bash -lc \
        "source /opt/ros/jazzy/setup.bash && python3 /ros2_ws/tools/write_motor_calibration.py"
Then check the result: python3 /ros2_ws/tools/read_motor_calibration.py (offsets/ranges unchanged),
restart the driver, and compare the arm with RViz.
"""
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/ttyACM0"
CALIBRATION_FILE = Path("/calibration/arm.json")


def main():
    robot = SO101Follower(SO101FollowerConfig(
        port=PORT, id=CALIBRATION_FILE.stem, calibration_dir=CALIBRATION_FILE.parent, use_radians=True))
    if not robot.calibration:
        raise SystemExit(f"No calibration loaded from {CALIBRATION_FILE}")
    robot.bus.connect()
    try:
        robot.bus.disable_torque()
        robot.bus.write_calibration(robot.calibration)
        print(f"Calibration of {CALIBRATION_FILE} written to the motors.")
        print("is_calibrated (motors == file):", robot.bus.is_calibrated)
    finally:
        robot.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
