#!/usr/bin/env python3
"""teleop_keyboard.py - drive the SO-ARM101 with the keyboard.

Run inside the control container (needs a terminal):
    source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash
    python3 /ros2_ws/teleop_keyboard.py

Keys (same physical position on AZERTY and QWERTY):
    e / d   shoulder_pan   + / -
    r / f   shoulder_lift  + / -
    t / g   elbow_flex     + / -
    y / h   wrist_flex     + / -
    u / j   wrist_roll     + / -
    i / k   gripper        open / close
    c / x   step bigger / smaller
    o       all joints to zero (gripper closed)
    Esc     quit
"""
import math
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER = "gripper"

# From so101_description/urdf (rad); gripper in % (LeRobot convention).
LIMITS = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    GRIPPER: (0.0, 100.0),
}
GRIPPER_RANGE_RAD = (-0.174533, 1.74533)  # to convert /joint_states back to %

KEYS = {  # key -> (joint, direction)
    "e": ("shoulder_pan", +1), "d": ("shoulder_pan", -1),
    "r": ("shoulder_lift", +1), "f": ("shoulder_lift", -1),
    "t": ("elbow_flex", +1), "g": ("elbow_flex", -1),
    "y": ("wrist_flex", +1), "h": ("wrist_flex", -1),
    "u": ("wrist_roll", +1), "j": ("wrist_roll", -1),
    "i": (GRIPPER, +1), "k": (GRIPPER, -1),
}
STEPS_RAD = [0.02, 0.05, 0.1, 0.2, 0.4]
GRIPPER_STEP_PCT = 10.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Teleop(Node):
    def __init__(self):
        super().__init__("teleop_keyboard")
        self.pub = self.create_publisher(JointState, "/joint_command", 10)
        self.sub = self.create_subscription(JointState, "/joint_states", self._on_state, 10)
        self.target = {j: 0.0 for j in ARM_JOINTS + [GRIPPER]}
        self.step_idx = 2
        self.synced = False

    def _on_state(self, msg):
        if self.synced:
            return
        for name, pos in zip(msg.name, msg.position):
            if name == GRIPPER:
                lo, hi = GRIPPER_RANGE_RAD
                self.target[name] = clamp((pos - lo) / (hi - lo) * 100.0, 0.0, 100.0)
            elif name in self.target:
                self.target[name] = pos
        self.synced = True

    def send(self, joints):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(joints)
        msg.position = [self.target[j] for j in joints]
        self.pub.publish(msg)

    def on_key(self, key):
        if key in KEYS:
            joint, sign = KEYS[key]
            step = GRIPPER_STEP_PCT if joint == GRIPPER else STEPS_RAD[self.step_idx]
            self.target[joint] = clamp(self.target[joint] + sign * step, *LIMITS[joint])
            self.send([joint])
        elif key == "c":
            self.step_idx = min(self.step_idx + 1, len(STEPS_RAD) - 1)
        elif key == "x":
            self.step_idx = max(self.step_idx - 1, 0)
        elif key == "o":
            for j in self.target:
                self.target[j] = 0.0
            self.send(list(self.target))

    def status(self):
        parts = [f"{j}={self.target[j]:+.2f}" for j in ARM_JOINTS]
        parts.append(f"gripper={self.target[GRIPPER]:3.0f}%")
        return f"step={STEPS_RAD[self.step_idx]:.2f}rad  " + "  ".join(parts)


def main():
    if not sys.stdin.isatty():
        sys.exit("teleop_keyboard needs a terminal: run it from `docker compose exec control bash`.")
    rclpy.init()
    node = Teleop()
    print(__doc__)

    # Start from the real position of the arm (wait up to 2 s for /joint_states).
    end = node.get_clock().now().nanoseconds + 2_000_000_000
    while not node.synced and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    print("synced with /joint_states" if node.synced else "no /joint_states received, starting from zero")

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            print("\r\033[K" + node.status(), end="", flush=True)
            if select.select([sys.stdin], [], [], 0.05)[0]:
                key = sys.stdin.read(1)
                if key in ("\x1b", "\x03"):  # Esc or Ctrl-C
                    break
                node.on_key(key.lower())
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
