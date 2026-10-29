#!/usr/bin/env python3
"""brain.launch.py - launches the brain of the arm (IK + pick & place) and the Ouija node
(phase 2: answers questions by moving the planchette; idle until a question arrives)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "ouija_layout",
            default_value="/opt/so101/sim/so101_sim/assets/so101/ouija/ouija_layout.json",
            description="letters / markers layout of the Ouija board (made by make_board.py)",
        ),
        DeclareLaunchArgument("max_joint_speed", default_value="1.0",
                              description="rad/s of the fastest joint of a motion (lower on the real arm)"),
        Node(
            package="so101_brain",
            executable="brain",
            name="so101_brain",
            output="screen",
            parameters=[{"max_joint_speed": ParameterValue(LaunchConfiguration("max_joint_speed"), value_type=float)}],
        ),
        Node(
            package="so101_brain",
            executable="ouija",
            name="so101_ouija",
            output="screen",
            parameters=[{"layout_file": LaunchConfiguration("ouija_layout")}],
        ),
    ])
