#!/usr/bin/env python3
"""wait_topic.py TOPIC [TIMEOUT_S] [TYPE] - exit 0 as soon as one message arrives on TOPIC, 1 otherwise.

Used by docker/run.sh to check that the perception node really publishes (ros2 topic hz is not
reliable for that from a script: output lost when killed by timeout, daemon discovery delays).
Default type sensor_msgs/msg/Image, best-effort subscription (matches reliable publishers too).

    python3 /ros2_ws/tools/wait_topic.py /perception/debug_image 25
"""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message


def main():
    topic = sys.argv[1]
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    msg_type = get_message(sys.argv[3] if len(sys.argv) > 3 else "sensor_msgs/msg/Image")
    rclpy.init()
    node = Node("wait_topic")
    got = []
    node.create_subscription(msg_type, topic, lambda m: got.append(1), qos_profile_sensor_data)
    t0 = time.monotonic()
    while not got and time.monotonic() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    print(f"{topic}: {'message received' if got else 'nothing'} after {time.monotonic() - t0:.1f} s")
    sys.exit(0 if got else 1)


if __name__ == "__main__":
    main()
