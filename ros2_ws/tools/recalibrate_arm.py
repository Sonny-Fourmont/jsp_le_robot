#!/usr/bin/env python3
"""recalibrate_arm.py - full calibration of the real arm (same procedure as `lerobot-calibrate`).

Interactive: it needs a terminal (docker compose exec WITHOUT -T) and your hands on the arm.
The driver must not hold the port: stop it first (docker compose stop control).

    docker compose run --rm --no-deps --entrypoint "" control bash -lc \
        "source /opt/ros/jazzy/setup.bash && python3 /ros2_ws/tools/recalibrate_arm.py"
(or `docker compose exec control ...` if the container is up and its driver is dead).

Steps (torque is OFF, the arm is limp: hold it):
  1. put ALL the joints in the MIDDLE of their range of motion (the pose of the LeRobot docs: arm
     upright, every joint centred), press ENTER -> writes the homing offsets (this defines the zero);
  2. move every joint except wrist_roll through its ENTIRE range, one after the other, press ENTER
     -> records range_min / range_max (wrist_roll is full turn: 0..4095);
  3. the file ros2_ws/calibration/so101_follower_<date>.json is written (and written in the motors).
Then copy it into calibration/ and set it as CALIBRATION_FILE in docker/.env.

`--ranges-only`: keeps the homing offsets already in the motors (step 1 done) and only redoes step 2
(ranges of motion), overwriting the file of the day. Use it when the recorded ranges were too small.
"""
import sys
import datetime
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/ttyACM0"
OUT_DIR = Path("/ros2_ws/calibration")


def main():
    name = f"so101_follower_{datetime.date.today().isoformat()}"
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id=name, calibration_dir=OUT_DIR, use_radians=True))
    if "--ranges-only" in sys.argv:
        ranges_only(robot)
        return
    if robot.calibration:
        print(f"{robot.calibration_fpath} already exists: it would be reused. Remove it first.")
        return
    robot.bus.connect()
    try:
        robot.calibrate()
    finally:
        robot.bus.disconnect(disable_torque=True)
    print(f"\nDone: {robot.calibration_fpath}")


def ranges_only(robot):
    from lerobot.motors import MotorCalibration

    robot.bus.connect()
    try:
        robot.bus.disable_torque()
        homing = {m: c.homing_offset for m, c in robot.bus.read_calibration().items()}
        print("Homing offsets kept:", homing)
        moving = [m for m in robot.bus.motors if m != "wrist_roll"]
        print("Move ALL joints except 'wrist_roll' from one end stop to the other, gripper included.\n"
              "Reference spreads (max - min): pan ~2500, lift ~2400, elbow ~2200, wrist_flex ~2300, gripper ~1500.\n"
              "Press ENTER to stop...")
        mins, maxes = robot.bus.record_ranges_of_motion(moving)
        mins["wrist_roll"], maxes["wrist_roll"] = 0, 4095
        robot.calibration = {m: MotorCalibration(id=robot.bus.motors[m].id, drive_mode=0, homing_offset=homing[m],
                                                 range_min=mins[m], range_max=maxes[m]) for m in robot.bus.motors}
        robot.bus.write_calibration(robot.calibration)
        robot._save_calibration()
        print(f"\nDone: {robot.calibration_fpath}")
    finally:
        robot.bus.disconnect(disable_torque=True)


if __name__ == "__main__":
    main()
