#!/usr/bin/env python3
"""perception.launch.py - launches the perception nodes:
  - camera_calib: camera pose (TF world -> camera) from the ArUco A4 sheet. With
    camera_tf:=sim it only evaluates itself against the simulator ground truth; with
    camera_tf:=calib it publishes the TF (the driver then does not), as on the real robot;
  - perception: phase 1, red cube + drop box -> 3D;
  - ouija_perception: phase 2, Ouija board (ArUco) + planchette -> 3D
    (idle if no board is visible, so both can always run).

Real robot (docker/.env -> docker-compose.yml): sheet_pose / board_pose are the measured
centres of the printed sheet / board ("x y yaw_deg"), params_file holds the HSV thresholds
tuned on the real camera (ros2_ws/config/perception_real.yaml, see tools/hsv_tune.py).
With USE_SIM=false, the sheet pose measured in ros2_ws/calibration/workspace.yaml
(sheet_pose: [x, y, z, yaw_deg], z = height of the table in the robot frame) takes
precedence over sheet_pose; the table plane, the drop-box colour and heights also come
from that file (read by perception_node.py)."""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

SHEET_Z = 0.001  # m, paper on the table (simulation: the table is z = 0)
BOARD_Z = 0.004  # m, Ouija board thickness (markers on its top face)
WORKSPACE_FILE = Path("/ros2_ws/calibration/workspace.yaml")


def _pose3(text, name):
    try:
        x, y, yaw = (float(v) for v in text.replace(",", " ").split())
    except ValueError as exc:
        raise ValueError(f"{name}: expected 'x y yaw_deg' (m, deg), got {text!r}") from exc
    return x, y, yaw


def _real_sheet_pose():
    """[x, y, z, yaw_deg] of the A4 sheet measured on the real arm (calibration/workspace.yaml),
    None in simulation or if the file has no sheet_pose."""
    if os.environ.get("USE_SIM", "true").lower() == "true" or not WORKSPACE_FILE.is_file():
        return None
    import yaml

    pose = (yaml.safe_load(WORKSPACE_FILE.read_text()) or {}).get("sheet_pose")
    if not pose:
        return None
    if len(pose) != 4:
        raise ValueError(f"{WORKSPACE_FILE}: sheet_pose must be [x, y, z, yaw_deg], got {pose!r}")
    return [float(v) for v in pose]


def _nodes(context):
    cfg = lambda key: LaunchConfiguration(key).perform(context)  # noqa: E731
    actions = []
    sheet = _real_sheet_pose()
    if sheet:
        sx, sy, sz, syaw = sheet  # measured on the real arm, z = table height in the robot frame
        actions.append(LogInfo(msg=f"camera_calib: A4 sheet pose from {WORKSPACE_FILE}: {sheet}"))
    else:
        sx, sy, syaw = _pose3(cfg("sheet_pose"), "sheet_pose")
        sz = SHEET_Z
    bx, by, byaw = _pose3(cfg("board_pose"), "board_pose")
    sheet_poses = [sx, sy, sz, syaw, bx, by, BOARD_Z, byaw]  # 4 values per sheet
    params_file = cfg("params_file").strip()
    # "none" because ros2 launch rejects an empty "params_file:=" (docker-compose default)
    extra = [params_file] if params_file and params_file.lower() not in ("none", "-") else []
    publish_tf = cfg("camera_tf") != "sim"

    return actions + [
        Node(
            package="so101_perception",
            executable="camera_calib",
            name="camera_calib",
            output="screen",
            parameters=[{
                "camera_topic": "/external_cam",
                "sheet_files": ParameterValue(LaunchConfiguration("calib_sheets")),  # yaml list
                "sheet_poses": sheet_poses,
                "mode": cfg("calib_mode"),
                "publish_tf": publish_tf,
            }],
        ),
        Node(
            package="so101_perception",
            executable="ouija_perception",
            name="ouija_perception",
            output="screen",
            parameters=[{
                "camera_topic": "/external_cam",
                "layout_file": cfg("ouija_layout"),
            }] + extra,
        ),
        Node(
            package="so101_perception",
            executable="perception",
            name="so101_perception",
            output="screen",
            parameters=[{
                "camera_topic": "/external_cam",
                # object_plane_z / box_plane_z / box HSV: defaults in perception_node.py (0.025 and an
                # orange box in simulation, calibration/workspace.yaml on the real arm)
            }] + extra,
        ),
    ]


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
            "sheet_pose", default_value="-0.15 0.20 0.0",
            description="centre of the A4 calibration sheet in the robot frame: 'x y yaw_deg' (m, deg); "
                        "matches scene_tek5.xml in simulation, measured on the real robot (.env CALIB_SHEET_POSE); "
                        "ignored when USE_SIM=false and calibration/workspace.yaml has a sheet_pose",
        ),
        DeclareLaunchArgument(
            "board_pose", default_value="0.235 0.0 0.0",
            description="centre of the Ouija board: 'x y yaw_deg'; matches scene_ouija.xml in simulation, "
                        "measured on the real robot (.env OUIJA_BOARD_POSE)",
        ),
        DeclareLaunchArgument(
            "params_file", default_value="none",
            description="optional yaml of parameters for perception / ouija_perception (HSV thresholds of "
                        "the real camera, ros2_ws/config/perception_real.yaml); none = built-in defaults",
        ),
        OpaqueFunction(function=_nodes),
    ])
