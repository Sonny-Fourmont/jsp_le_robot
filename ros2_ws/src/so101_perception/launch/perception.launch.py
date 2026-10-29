#!/usr/bin/env python3
"""perception.launch.py - launches the perception nodes:
  - camera_calib: camera pose (TF world -> camera) from the ArUco A4 sheet. With
    camera_tf:=sim it only evaluates itself against the simulator ground truth; with
    camera_tf:=calib it publishes the TF (the driver then does not), as on the real robot;
  - perception: phase 1, red cube + drop box -> 3D;
  - ouija_perception: phase 2, Ouija board (ArUco) + planchette -> 3D
    (idle if no board is visible, so both can always run)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "ouija_layout",
            default_value="/opt/so101/sim/so101_sim/assets/so101/ouija/ouija_layout.json",
            description="letters / markers layout of the Ouija board (made by make_board.py)",
        ),
        DeclareLaunchArgument("camera_tf", default_value="sim",
                              description="sim: camera TF from the simulator; calib: from camera_calib (ArUco sheet)"),
        DeclareLaunchArgument("calib_mode", default_value="calibrate",
                              description="camera_calib: calibrate (sheet visible) or load (saved calib_file)"),
        DeclareLaunchArgument(
            "calib_sheets",
            default_value="[/opt/so101/sim/so101_sim/assets/so101/calib/calib_sheet.json, "
                          "/opt/so101/sim/so101_sim/assets/so101/ouija/ouija_layout.json]",
            description="marker layouts usable for the calibration: the A4 sheet (make_calib_sheet.py) "
                        "and the Ouija board (make_board.py); whichever is seen is used",
        ),
        DeclareLaunchArgument(
            "calib_poses", default_value="[-0.15, 0.20, 0.001, 0.0, 0.235, 0.0, 0.004, 0.0]",
            description="centre of each sheet in the robot frame: x, y, z (m), yaw (deg), 4 per sheet; "
                        "must match scene_tek5.xml / scene_ouija.xml in simulation",
        ),
        Node(
            package="so101_perception",
            executable="camera_calib",
            name="camera_calib",
            output="screen",
            parameters=[{
                "camera_topic": "/external_cam",
                "sheet_files": ParameterValue(LaunchConfiguration("calib_sheets")),  # yaml lists
                "sheet_poses": ParameterValue(LaunchConfiguration("calib_poses")),
                "mode": LaunchConfiguration("calib_mode"),
                "publish_tf": ParameterValue(
                    PythonExpression(["'", LaunchConfiguration("camera_tf"), "' != 'sim'"]), value_type=bool),
            }],
        ),
        Node(
            package="so101_perception",
            executable="ouija_perception",
            name="ouija_perception",
            output="screen",
            parameters=[{
                "camera_topic": "/external_cam",
                "layout_file": LaunchConfiguration("ouija_layout"),
            }],
        ),
        Node(
            package="so101_perception",
            executable="perception",
            name="so101_perception",
            output="screen",
            parameters=[{
                "camera_topic": "/external_cam",
                "object_plane_z": 0.025,
                "box_plane_z": 0.025,
            }],
        ),
    ])
