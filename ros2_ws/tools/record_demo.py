#!/usr/bin/env python3
"""record_demo.py - records a video of the pick & place in simulation, without a screen.

Layouts:
  (default)  three panels: 3D view of the simulation (MuJoCo replay of /joint_states and of the
             cube's ground truth, rendered off-screen), the overhead camera (/external_cam/image_raw)
             and the perception debug image (/perception/debug_image).
  --main     the 3D MuJoCo view only, full frame 1280x720.
  --rviz     RViz full frame 1280x720: rviz2 runs on a virtual X server (Xvfb) inside the container
             and its window is grabbed with ffmpeg (x11grab). Shows the URDF model, the TF frame of
             the gripper, the FK position published by the brain (/end_effector_markers), the
             perception marker and the ground truth of the simulation (/debug/sim_markers). The
             caption compares the TCP position given by our FK (pinocchio, /end_effector_pose) with
             the TF of robot_state_publisher (same URDF, independent computation).
  --grid     everything the teacher asks for in one 1920x1080 frame: the MuJoCo scene, RViz (URDF,
             TF, FK marker, with the FK vs TF figures), the camera and the perception image, plus
             the caption with the current step.
Runs N randomized trials (/sim/randomize_object + /pick_and_place), checked with the ground truth
like trial_pick_place.py. /brain/state is the caption.
Usage: record_demo.py [out.mp4] [N] [--main|--rviz|--grid]

    docker compose exec control bash -lc "source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash \
        && python3 /ros2_ws/tools/record_demo.py /ros2_ws/rec/pick_place_sim.mp4 3"
"""
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

ASSETS = Path("/opt/so101/sim/so101_sim/assets/so101")
RVIZ_CONFIG = os.environ.get("RVIZ_CONFIG", "/ros2_ws/so101_video.rviz")
GRID_RVIZ_CONFIG = os.environ.get("RVIZ_CONFIG", "/ros2_ws/so101_grid.rviz")  # same, framed for a 960x494 panel
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
URDF_GRIPPER_RANGE = (-0.174533, 1.74533)  # gripper in /joint_states (rad, URDF), see driver_node
TCP_FRAME = "gripper_frame_link"
FPS = 20
VIEW_W, VIEW_H = 960, 720  # 3D view of the panels layout (scene offscreen buffer is 1280x720)
MAIN_W, MAIN_H = 1280, 720  # --main / --rviz: one view, full frame
SIDE_W, SIDE_H = 480, 360  # camera + perception panels
GRID_W, GRID_H = 1920, 1080  # --grid: title band, 2x2 panels of PW x PH, caption bar
PW, PH = 960, 494
GRID_TITLE_H, GRID_BAR_H = 40, 52
FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLES = {
    "panels": "SO-ARM101  pick & place en simulation (MuJoCo + ROS 2)",
    "main": "SO-ARM101  pick & place en simulation (MuJoCo + ROS 2)",
    "rviz": "SO-ARM101  pick & place vu dans RViz : modele URDF + TF, position du TCP par FK",
    "grid": "SO-ARM101  pick & place en simulation : scene MuJoCo, RViz (URDF + TF + FK), camera, detection",
}


def img_from_msg(msg):
    """sensor_msgs/Image -> BGR numpy (rgb8, bgr8, mono8)."""
    buf = np.frombuffer(msg.data, np.uint8)
    if msg.encoding == "mono8":
        return cv2.cvtColor(buf.reshape(msg.height, msg.step)[:, : msg.width], cv2.COLOR_GRAY2BGR)
    img = buf.reshape(msg.height, msg.step // 3, 3)[:, : msg.width]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else img.copy()


def text(img, s, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def panel(img, w, h, label, fit=False):
    """Image resized to w x h under a label band; fit=True keeps the aspect ratio (black bars)."""
    if img is None:
        out = np.full((h, w, 3), 40, np.uint8)
        text(out, "en attente...", (w // 2 - 90, h // 2), 0.8, (160, 160, 160))
    elif fit and abs(img.shape[1] / img.shape[0] - w / h) > 0.02:
        k = min(w / img.shape[1], h / img.shape[0])
        rw, rh = int(img.shape[1] * k), int(img.shape[0] * k)
        out = np.zeros((h, w, 3), np.uint8)
        x0, y0 = (w - rw) // 2, (h - rh) // 2
        out[y0:y0 + rh, x0:x0 + rw] = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    else:
        out = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(out, (0, 0), (w, 34), (20, 20, 20), -1)
    text(out, label, (10, 24), 0.6, (240, 240, 240), 1)
    return out


class DemoNode(Node):
    def __init__(self):
        super().__init__("record_demo")
        self.lock = threading.Lock()
        self.q = {}
        self.gt_o = self.gt_b = None
        self.fk = None  # TCP position from the brain's FK (/end_effector_pose)
        self.cam = self.dbg = None
        self.state = "IDLE"  # /brain/state is only published on change
        self.trial = (0, 0)
        self.verdict = ""
        best_effort = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, "/joint_states", self._cb_js, 10)
        self.create_subscription(PoseStamped, "/debug/object_position_gt", lambda m: self._set_pos("gt_o", m), 5)
        self.create_subscription(PoseStamped, "/debug/drop_box_position_gt", lambda m: self._set_pos("gt_b", m), 5)
        self.create_subscription(PoseStamped, "/end_effector_pose", lambda m: self._set_pos("fk", m), 5)
        self.create_subscription(Image, "/external_cam/image_raw", lambda m: self._set_img("cam", m), best_effort)
        self.create_subscription(Image, "/perception/debug_image", lambda m: self._set_img("dbg", m), best_effort)
        self.create_subscription(String, "/brain/state", self._cb_state, 5)
        self.randomize = self.create_client(Trigger, "/sim/randomize_object")
        self.pick = self.create_client(Trigger, "/pick_and_place")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _cb_js(self, msg):
        with self.lock:
            self.q = dict(zip(msg.name, msg.position))

    def _set_pos(self, key, msg):
        p = msg.pose.position
        with self.lock:
            setattr(self, key, np.array([p.x, p.y, p.z]))

    def _set_img(self, key, msg):
        img = img_from_msg(msg)
        with self.lock:
            setattr(self, key, img)

    def _cb_state(self, msg):
        with self.lock:
            self.state = msg.data

    def tcp_from_tf(self):
        """TCP position from the TF tree (robot_state_publisher), None if not available yet."""
        try:
            t = self.tf_buffer.lookup_transform("world", TCP_FRAME, rclpy.time.Time()).transform.translation
            return np.array([t.x, t.y, t.z])
        except Exception:  # noqa: BLE001 - no TF yet
            return None

    def call(self, client):
        fut = client.call_async(Trigger.Request())
        while not fut.done():
            time.sleep(0.05)
        return fut.result()


class MujocoView:
    """Off-screen MuJoCo rendering of the scene, replaying /joint_states and the cube's ground truth."""

    def __init__(self, node, w, h, distance, lookat=(0.13, -0.04, 0.04)):
        import mujoco

        self.node, self.mujoco = node, mujoco
        scene = ASSETS / (os.environ.get("SIM_SCENE") or "scene_tek5.xml")
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=h, width=w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, 160.0, -22.0
        self.qadr = {j: self.model.joint(j).qposadr[0] for j in ARM_JOINTS + ["gripper"]}
        self.cube = self.model.joint("pickup_object_free").qposadr[0]
        self.g_range = tuple(float(v) for v in self.model.joint("gripper").range)

    def frame(self, q, gt_o):
        for j in ARM_JOINTS:
            if j in q:
                self.data.qpos[self.qadr[j]] = q[j]
        if "gripper" in q:
            u_lo, u_hi = URDF_GRIPPER_RANGE
            g_lo, g_hi = self.g_range
            pct = np.clip((q["gripper"] - u_lo) / (u_hi - u_lo), 0.0, 1.0)
            self.data.qpos[self.qadr["gripper"]] = g_lo + pct * (g_hi - g_lo)
        if gt_o is not None:
            self.data.qpos[self.cube : self.cube + 3] = gt_o
            self.data.qpos[self.cube + 3 : self.cube + 7] = (1.0, 0.0, 0.0, 0.0)
        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam)
        # software (osmesa) rendering: shadows and reflections cost 4x, 215 -> 47 ms per frame
        self.renderer.scene.flags[self.mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.renderer.scene.flags[self.mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)

    def close(self):
        self.renderer.close()


class RvizView:
    """rviz2 in fullscreen on a virtual X server (Xvfb), grabbed with ffmpeg at FPS."""

    def __init__(self, w, h, config, fps, log=Path("/tmp/record_rviz.log")):
        self.w, self.h = w, h
        self.latest = None
        self.lock = threading.Lock()
        self.log = open(log, "w")
        for tool in ("Xvfb", "rviz2", "ffmpeg"):
            if shutil.which(tool) is None:
                raise RuntimeError(f"{tool} not found in the container (rebuild the control image: docker compose build control)")
        # /tmp/.X11-unix is shared with the host (compose): take a display number far from the host's
        self.display = next((f":{n}" for n in range(90, 100) if not self._display_busy(n)), None)
        if self.display is None:
            raise RuntimeError("no free X display between :90 and :99")
        self.xvfb = subprocess.Popen(
            ["Xvfb", self.display, "-screen", "0", f"{w}x{h}x24", "-nolisten", "tcp", "-ac", "-dpi", "96"],
            stdout=self.log, stderr=self.log)
        for _ in range(50):
            if Path(f"/tmp/.X11-unix/X{self.display[1:]}").exists():
                break
            if self.xvfb.poll() is not None:
                raise RuntimeError("Xvfb did not start (see /tmp/record_rviz.log)")
            time.sleep(0.1)
        env = dict(os.environ, DISPLAY=self.display, QT_QPA_PLATFORM="xcb", LIBGL_ALWAYS_SOFTWARE="1",
                   QT_X11_NO_MITSHM="1", XDG_RUNTIME_DIR="/tmp")
        self.rviz = subprocess.Popen(["rviz2", "--fullscreen", "-d", config], env=env, stdout=self.log, stderr=self.log)
        self.ffmpeg = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-f", "x11grab", "-draw_mouse", "0", "-framerate", str(fps),
             "-video_size", f"{w}x{h}", "-i", self.display, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    @staticmethod
    def _display_busy(n):
        if Path(f"/tmp/.X11-unix/X{n}").exists() or Path(f"/tmp/.X{n}-lock").exists():
            return True
        for proc in Path("/proc").glob("[0-9]*/cmdline"):  # an Xvfb left behind (abstract socket, no file)
            try:
                if f"Xvfb\0:{n}\0".encode() in proc.read_bytes():
                    return True
            except OSError:
                pass
        return False

    def _read(self):
        n = self.w * self.h * 3
        buf = bytearray(n)
        while True:
            view, got = memoryview(buf), 0
            while got < n:  # unbuffered pipe: short reads
                k = self.ffmpeg.stdout.readinto(view[got:])
                if not k:
                    return
                got += k
            frame = np.frombuffer(bytes(buf), np.uint8).reshape(self.h, self.w, 3)
            with self.lock:
                self.latest = frame

    def wait_ready(self, timeout=90.0):
        """Wait until RViz draws something (non-black frame), then a little more for the robot model."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            f = self.frame()
            if self.rviz.poll() is not None:
                raise RuntimeError(f"rviz2 exited with code {self.rviz.returncode} (see /tmp/record_rviz.log)")
            if f is not None and f.mean() > 5:
                time.sleep(4.0)
                return
            time.sleep(0.5)
        got = "no frame received from ffmpeg" if self.frame() is None else "only black frames"
        raise RuntimeError(f"RViz did not draw anything in time ({got}, see /tmp/record_rviz.log)")

    def frame(self, *_):
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def close(self):
        for proc in (self.ffmpeg, self.rviz, self.xvfb):
            if proc.poll() is None:
                proc.terminate()
        for proc in (self.ffmpeg, self.rviz, self.xvfb):
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
        for f in (f"/tmp/.X11-unix/X{self.display[1:]}", f"/tmp/.X{self.display[1:]}-lock"):
            Path(f).unlink(missing_ok=True)  # Xvfb leaves them behind when terminated
        self.log.close()


class VideoWriter:
    """H.264 through ffmpeg when available (plays everywhere, 3x smaller), else OpenCV mp4v."""

    def __init__(self, out, fps, w, h):
        self.proc = self.cv = None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                 "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame):
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)

    def release(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


class Recorder(threading.Thread):
    def __init__(self, node, view, out, fps, layout, rviz=None):
        super().__init__(daemon=True)
        self.node, self.view, self.out, self.fps, self.layout = node, view, out, fps, layout
        self.rviz = rviz  # --grid: RvizView next to the MuJoCo view
        self.stop = threading.Event()
        self.n = 0
        self.t0 = time.monotonic()

    @staticmethod
    def _fk_box(frame, fk, tf):
        """Top-left inset: TCP position by our FK vs the TF tree (two independent computations)."""
        x0, y0 = 16, 52
        cv2.rectangle(frame, (x0 - 6, y0), (x0 + 760, y0 + 92), (20, 20, 20), -1)
        text(frame, "Position du TCP (gripper_frame_link)", (x0, y0 + 24), 0.6, (255, 220, 120), 1)
        fmt = lambda v: f"x {v[0]:+.3f}   y {v[1]:+.3f}   z {v[2]:+.3f} m" if v is not None else "..."  # noqa: E731
        text(frame, "FK pinocchio  /end_effector_pose", (x0, y0 + 52), 0.55, (160, 230, 255), 1)
        text(frame, fmt(fk), (x0 + 400, y0 + 52), 0.55, (160, 230, 255), 1)
        text(frame, "TF robot_state_publisher (URDF)", (x0, y0 + 80), 0.55, (160, 230, 255), 1)
        text(frame, fmt(tf), (x0 + 400, y0 + 80), 0.55, (160, 230, 255), 1)
        if fk is not None and tf is not None:
            text(frame, f"ecart {np.linalg.norm(fk - tf) * 1000:.1f} mm", (x0 + 640, y0 + 24), 0.6, (120, 255, 150), 1)

    @staticmethod
    def _fk_lines(frame, fk, tf, x0, y0):
        """Compact FK vs TF comparison (two lines), drawn inside the RViz panel of the grid."""
        cv2.rectangle(frame, (x0 - 6, y0 - 22), (x0 + 640, y0 + 32), (20, 20, 20), -1)
        fmt = lambda v: f"x {v[0]:+.3f}  y {v[1]:+.3f}  z {v[2]:+.3f} m" if v is not None else "..."  # noqa: E731
        text(frame, f"TCP par FK (pinocchio, /end_effector_pose)  {fmt(fk)}", (x0, y0), 0.5, (160, 230, 255), 1)
        gap = f"   ecart {np.linalg.norm(fk - tf) * 1000:.1f} mm" if fk is not None and tf is not None else ""
        text(frame, f"TCP par TF (robot_state_publisher, URDF)    {fmt(tf)}{gap}", (x0, y0 + 24), 0.5,
             (160, 230, 255), 1)

    def _run_grid(self):
        """--grid: MuJoCo scene | RViz over camera | perception, title band above, caption bar below."""
        writer = VideoWriter(self.out, self.fps, GRID_W, GRID_H)
        y1, y2 = GRID_TITLE_H, GRID_TITLE_H + PH
        t_start = time.monotonic()
        while not self.stop.is_set():
            n = self.node
            with n.lock:
                q, gt_o, fk, cam_img, dbg_img = dict(n.q), n.gt_o, n.fk, n.cam, n.dbg
                state, trial, verdict = n.state, n.trial, n.verdict
            frame = np.zeros((GRID_H, GRID_W, 3), np.uint8)
            frame[y1:y2, :PW] = panel(self.view.frame(q, gt_o), PW, PH, "Simulation MuJoCo (verite terrain)")
            frame[y1:y2, PW:] = panel(self.rviz.frame(), PW, PH,
                                      "RViz : modele URDF, TF de la pince, FK du TCP (/end_effector_markers)")
            frame[y2:y2 + PH, :PW] = panel(cam_img, PW, PH, "Camera du dessus  /external_cam/image_raw", fit=True)
            frame[y2:y2 + PH, PW:] = panel(dbg_img, PW, PH, "Perception  /perception/debug_image  (cube + boite -> 3D)",
                                           fit=True)
            self._fk_lines(frame, fk, n.tcp_from_tf(), PW + 12, y1 + 34 + 30)
            text(frame, TITLES["grid"], (16, 30), 0.8)
            cv2.rectangle(frame, (0, GRID_H - GRID_BAR_H), (GRID_W, GRID_H), (20, 20, 20), -1)
            t = time.monotonic() - self.t0
            head = f"Essai {trial[0]}/{trial[1]}   " if trial[1] else ""
            text(frame, f"{head}Etat : {state}   {t:5.1f} s", (16, GRID_H - 18), 0.75, (255, 220, 120))
            if verdict:
                text(frame, verdict, (PW, GRID_H - 18), 0.75, (120, 255, 150))
            due = int((time.monotonic() - t_start) * self.fps) + 1
            for _ in range(max(1, due - self.n)):
                writer.write(frame)
                self.n += 1
            time.sleep(max(0.0, t_start + self.n / self.fps - time.monotonic()))
        writer.release()

    def run(self):
        if self.layout == "grid":
            return self._run_grid()
        one_view = self.layout in ("main", "rviz")
        view_w, view_h = (MAIN_W, MAIN_H) if one_view else (VIEW_W, VIEW_H)
        frame_w = view_w if one_view else VIEW_W + SIDE_W
        bar = 92  # caption bar height
        writer = VideoWriter(self.out, self.fps, frame_w, view_h)
        t_start = time.monotonic()
        while not self.stop.is_set():
            n = self.node
            with n.lock:
                q, gt_o, fk, cam_img, dbg_img = dict(n.q), n.gt_o, n.fk, n.cam, n.dbg
                state, trial, verdict = n.state, n.trial, n.verdict
            view = self.view.frame(q, gt_o)
            if view is None:
                view = np.full((view_h, view_w, 3), 40, np.uint8)
            if one_view:
                frame = view
            else:
                frame = np.zeros((VIEW_H, VIEW_W + SIDE_W, 3), np.uint8)
                frame[:, :VIEW_W] = view
                frame[:SIDE_H, VIEW_W:] = panel(cam_img, SIDE_W, SIDE_H, "Camera du dessus  /external_cam/image_raw")
                frame[SIDE_H:, VIEW_W:] = panel(dbg_img, SIDE_W, SIDE_H, "Perception  /perception/debug_image")
            text(frame, TITLES[self.layout], (16, 36), 0.8)
            cv2.rectangle(frame, (0, view_h - bar), (view_w, view_h), (20, 20, 20), -1)
            t = time.monotonic() - self.t0
            head = f"Essai {trial[0]}/{trial[1]}   " if trial[1] else ""
            text(frame, f"{head}Etat : {state}   {t:5.1f} s", (16, view_h - bar + 36), 0.8, (255, 220, 120))
            if self.layout == "rviz":
                self._fk_box(frame, fk, n.tcp_from_tf())
            if verdict:
                text(frame, verdict, (16, view_h - 18), 0.8, (120, 255, 150))
            # real-time pacing: duplicate the frame if rendering fell behind, wait if ahead
            due = int((time.monotonic() - t_start) * self.fps) + 1
            for _ in range(max(1, due - self.n)):
                writer.write(frame)
                self.n += 1
            time.sleep(max(0.0, t_start + self.n / self.fps - time.monotonic()))
        writer.release()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    layout = ("grid" if "--grid" in sys.argv else "rviz" if "--rviz" in sys.argv
              else "main" if "--main" in sys.argv else "panels")
    out = args[0] if args else "/ros2_ws/rec/pick_place_sim.mp4"
    n_trials = int(args[1]) if len(args) > 1 else 3
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = DemoNode()
    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    if not (node.randomize.wait_for_service(20) and node.pick.wait_for_service(20)):
        print("services /sim/randomize_object and /pick_and_place not available")
        return
    deadline = time.time() + 20
    while time.time() < deadline and not (node.q and node.cam is not None and node.gt_o is not None
                                          and node.dbg is not None):
        time.sleep(0.2)
    if not node.q or node.cam is None or node.gt_o is None:
        print("no /joint_states, camera image or ground truth yet")
        return
    if node.dbg is None:
        print("no /perception/debug_image: the perception node receives nothing, docker compose restart perception")
        return

    rviz = None
    if layout == "rviz":
        view = RvizView(MAIN_W, MAIN_H, RVIZ_CONFIG, FPS)
    elif layout == "grid":
        view = MujocoView(node, PW, PH, 0.80, lookat=(0.13, -0.04, 0.10))
        rviz = RvizView(PW, PH, GRID_RVIZ_CONFIG, FPS)
    elif layout == "main":
        view = MujocoView(node, MAIN_W, MAIN_H, 0.66)
    else:
        view = MujocoView(node, VIEW_W, VIEW_H, 0.72)

    try:
        for v in (view, rviz):
            if isinstance(v, RvizView):
                print(f"rviz2 on virtual display {v.display}, waiting for the first frame...", flush=True)
                v.wait_ready()
        ok = run_trials(node, view, out, n_trials, layout, rviz)
    finally:
        view.close()
        if rviz is not None:
            rviz.close()
    rclpy.shutdown()


def run_trials(node, view, out, n_trials, layout, rviz=None):
    rec = Recorder(node, view, out, FPS, layout, rviz)
    rec.start()
    time.sleep(1.5)
    ok, lines = 0, []
    for i in range(n_trials):
        with node.lock:
            node.trial, node.verdict = (i + 1, n_trials), "cube deplace au hasard"
        node.call(node.randomize)
        time.sleep(1.5)
        start = node.gt_o.copy()
        with node.lock:
            node.verdict = ""
        t0 = time.time()
        res = node.call(node.pick)
        dt = time.time() - t0
        time.sleep(1.0)
        o, b = node.gt_o, node.gt_b
        in_box = bool(np.linalg.norm((o - b)[:2]) < 0.05 and o[2] < 0.06)
        ok += in_box
        verdict = "cube dans la boite" if in_box else "cube HORS de la boite"
        with node.lock:
            node.verdict = f"Essai {i + 1} : {verdict} ({dt:.0f} s, verifie par la verite terrain)"
        r, a = np.hypot(*start[:2]), np.degrees(np.arctan2(start[1], start[0]))
        lines.append(f"trial {i + 1}: start r={r:.2f} a={a:.0f}deg -> {res.success} in {dt:.0f}s, {verdict}  [{res.message[:80]}]")
        print(lines[-1], flush=True)
        time.sleep(3.0)
    with node.lock:
        node.verdict = f"Bilan : {ok}/{n_trials} cubes dans la boite"
    time.sleep(3.0)
    rec.stop.set()
    rec.join()
    print(f"wrote {out}: {rec.n} frames, {rec.n / FPS:.0f} s, success {ok}/{n_trials}")
    return ok


if __name__ == "__main__":
    main()
