#!/usr/bin/env python3
"""driver.launch.py - launches the SO-ARM101 driver node

    ros2 launch so101_driver driver.launch.py                # simulation
    ros2 launch so101_driver driver.launch.py use_sim:=false # real robot
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim", default_value="true",
                              description="Simulated backend or real robot"),
        DeclareLaunchArgument("port", default_value="/dev/ttyACM0",  #  Use the following command to find the robot port:
                              # sudo udevadm monitor --subsystem-match=tty --property | grep --line-buffered -E "(DEVNAME=|ACTION=)"
                              description="Serial port of the real SO-ARM101"),
        DeclareLaunchArgument("scene", default_value="",
                              description="Simulation scene file (sim assets dir), empty = default"),
        DeclareLaunchArgument("torque", default_value="true",
                              description="Real arm: false = motors limp (read only, move by hand)"),
        DeclareLaunchArgument("max_step", default_value="0.0",
                              description="Cap on each command: +/- rad around the present position (0 = off)"),
        DeclareLaunchArgument("camera_tf", default_value="sim",
                              description="sim: camera TF from the simulator (ground truth); "
                              "calib: from the camera_calib node (ArUco sheet), as on the real robot"),
        Node(
            package="so101_driver",
            executable="driver",
            name="so101_driver",
            output="screen",
            parameters=[{
                "use_sim": ParameterValue(LaunchConfiguration("use_sim"), value_type=bool),
                "port": LaunchConfiguration("port"),
                "scene": LaunchConfiguration("scene"),
                "torque": ParameterValue(LaunchConfiguration("torque"), value_type=bool),
                "max_step": ParameterValue(LaunchConfiguration("max_step"), value_type=float),
                "publish_camera_tf": ParameterValue(
                    PythonExpression(["'", LaunchConfiguration("camera_tf"), "' == 'sim'"]), value_type=bool),
            }],
        ),
    ])
