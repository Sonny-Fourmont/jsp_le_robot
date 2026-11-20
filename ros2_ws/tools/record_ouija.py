#!/usr/bin/env python3
"""record_ouija.py - records a video of the Ouija demo in simulation, without a screen.

3D view of the simulation (MuJoCo replay of /joint_states and of the planchette's ground truth,
rendered off-screen, 1280x720) with the Ouija perception image as an inset (/ouija/debug_image:
board, ring, letter under the ring) and a caption: question, letters spelled so far (/ouija/state),
letter seen under the ring by the perception (/ouija/pointed). The answer is spelled with
/ouija/spell (the word is given, no LLM), so the video is predictable.

--real (automatic when USE_SIM=false): no simulation to render. RViz runs on a virtual X server
inside the container (robot model, TF of the gripper, FK marker, letters of the board seen by the
perception, current target) next to the real camera image and the Ouija perception image, with the
same caption. Nothing else to do on the real robot: the stack runs as usual (./run.sh video-reel).
--ask: the question goes through /ouija/ask (LLM, ollama) instead of spelling the given answer.
Usage: record_ouija.py [out.mp4] ["question"] [ANSWER] [--real] [--ask]

    docker compose exec control bash -lc "source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash \
        && python3 /ros2_ws/tools/record_ouija.py /ros2_ws/rec/ouija_demo.mp4 'Qui est le meilleur groupe ?' NOUS"
"""
import os
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
from so101_interfaces.srv import Ask
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_demo import (ARM_JOINTS, ASSETS, FPS, MAIN_H, MAIN_W, TCP_FRAME, URDF_GRIPPER_RANGE,  # noqa: E402
                         Recorder as DemoRecorder, RvizView, VideoWriter, img_from_msg, panel, text)

TITLE = "SO-ARM101  planche Ouija en simulation (MuJoCo + ROS 2)"
INSET_W, INSET_H = 400, 300
# over the robot's shoulder: the letters of the board are printed to be read from the robot's side
CAM_LOOKAT, CAM_DISTANCE, CAM_AZIMUTH, CAM_ELEVATION = (0.25, 0.03, 0.0), 0.9, 38.0, -52.0
# --real: RViz panel (robot, TF, FK, board letters) + camera + detection, 1920x1080
REAL_RVIZ_CONFIG = os.environ.get("RVIZ_CONFIG", "/ros2_ws/so101_real.rviz")
TITLE_REAL = "SO-ARM101  planche Ouija sur le vrai robot : RViz (URDF + TF + FK), camera, detection"
R_W, R_H, R_TITLE, R_BAR = 1920, 1080, 40, 126
RV_W, RV_H = 1312, R_H - R_TITLE - R_BAR  # RViz panel 1312 x 914
SIDE_W, SIDE_H = R_W - RV_W, (R_H - R_TITLE - R_BAR) // 2  # camera and detection panels 608 x 457


def strip_accents(s):
    """OpenCV's Hershey fonts have no accents."""
    table = str.maketrans("àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ«»’", "aaaeeeeiioouuucAAAEEEEIIOOUUUC\"\"'")
    return s.translate(table)


class OuijaDemoNode(Node):
    def __init__(self):
        super().__init__("record_ouija")
        self.lock = threading.Lock()
        self.q = {}
        self.gt_puck = self.gt_cube = None
        self.dbg = self.cam = None
        self.fk = None  # TCP position from the brain's FK (/end_effector_pose)
        self.state, self.pointed, self.answer = "IDLE", "", ""
        self.spelled = []  # one entry per DWELL on a letter
        self.verdict = ""
        best_effort = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, "/joint_states", self._cb_js, 10)
        self.create_subscription(PoseStamped, "/debug/planchette_position_gt", lambda m: self._set_pos("gt_puck", m), 5)
        self.create_subscription(PoseStamped, "/debug/object_position_gt", lambda m: self._set_pos("gt_cube", m), 5)
        self.create_subscription(Image, "/ouija/debug_image", lambda m: self._cb_img("dbg", m), best_effort)
        self.create_subscription(Image, "/external_cam/image_raw", lambda m: self._cb_img("cam", m), best_effort)
        self.create_subscription(PoseStamped, "/end_effector_pose", lambda m: self._set_pos("fk", m), 5)
        self.create_subscription(String, "/ouija/state", self._cb_state, 5)
        self.create_subscription(String, "/ouija/pointed", self._cb_pointed, 5)
        self.create_subscription(String, "/ouija/answer", self._cb_answer, 5)
        self.spell = self.create_client(Ask, "/ouija/spell")
        self.ask = self.create_client(Ask, "/ouija/ask")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _cb_js(self, msg):
        with self.lock:
            self.q = dict(zip(msg.name, msg.position))

    def _set_pos(self, key, msg):
        p = msg.pose.position
        with self.lock:
            setattr(self, key, np.array([p.x, p.y, p.z]))

    def _cb_img(self, key, msg):
        img = img_from_msg(msg)
        with self.lock:
            setattr(self, key, img)

    def tcp_from_tf(self):
        """TCP position from the TF tree (robot_state_publisher), None if not available yet."""
        try:
            t = self.tf_buffer.lookup_transform("world", TCP_FRAME, rclpy.time.Time()).transform.translation
            return np.array([t.x, t.y, t.z])
        except Exception:  # noqa: BLE001 - no TF yet
            return None

    def _cb_state(self, msg):
        with self.lock:
            self.state = msg.data
            parts = msg.data.split()
            if len(parts) >= 2 and parts[0] == "DWELL" and parts[1] != "FIN":
                self.spelled.append(parts[1])

    def _cb_pointed(self, msg):
        with self.lock:
            self.pointed = msg.data

    def _cb_answer(self, msg):
        with self.lock:
            self.answer = msg.data


class Recorder(threading.Thread):
    def __init__(self, node, out, question, answer, layout="sim", rviz=None):
        super().__init__(daemon=True)
        self.node, self.out, self.question, self.answer = node, out, question, answer
        self.layout, self.rviz = layout, rviz
        self.stop = threading.Event()
        self.n = 0

    def _caption(self, frame, w, h, t, state, pointed, spelled, verdict, right_x):
        """Caption bar (R_BAR px): state, question, letters spelled so far, letter seen under the ring."""
        bar = R_BAR
        cv2.rectangle(frame, (0, h - bar), (w, h), (20, 20, 20), -1)
        text(frame, f"Etat : {state}   {t:5.1f} s", (16, h - bar + 34), 0.8, (255, 220, 120))
        target = self.answer or self.node.answer  # OUI / NON / FIN are one target of the board, not letters
        expected = 1 if target.upper() in ("OUI", "NON", "FIN") else len(target)
        shown = " ".join(spelled) + (" _" * max(0, expected - len(spelled)) if not verdict else "")
        text(frame, f"Question : {strip_accents(self.question)}", (16, h - bar + 68), 0.7, (255, 255, 255))
        text(frame, f"Reponse epelee : {shown.strip()}", (16, h - bar + 102), 0.8, (120, 255, 150))
        if pointed:
            text(frame, f"Lettre sous l'anneau (perception) : {pointed}", (right_x, h - bar + 68), 0.7, (160, 230, 255))
        if verdict:
            text(frame, verdict, (right_x, h - bar + 102), 0.7, (120, 255, 150))

    def _run_real(self):
        """--real: RViz | camera over detection, title band above, caption bar below (1920x1080)."""
        writer = VideoWriter(self.out, FPS, R_W, R_H)
        y1 = R_TITLE
        t_start = time.monotonic()
        while not self.stop.is_set():
            n = self.node
            with n.lock:
                cam, dbg, fk = n.cam, n.dbg, n.fk
                state, pointed, spelled, verdict = n.state, n.pointed, list(n.spelled), n.verdict
            frame = np.zeros((R_H, R_W, 3), np.uint8)
            rv = self.rviz.frame() if self.rviz is not None else None
            if rv is None and self.rviz is None:
                rv = np.full((RV_H, RV_W, 3), 40, np.uint8)
                text(rv, "RViz indisponible (voir /tmp/record_rviz.log)", (RV_W // 2 - 330, RV_H // 2), 0.8, (160, 160, 160))
            frame[y1:y1 + RV_H, :RV_W] = panel(rv, RV_W, RV_H,
                                               "RViz : modele URDF, TF de la pince, FK du TCP, lettres de la planche (perception)")
            frame[y1:y1 + SIDE_H, RV_W:] = panel(cam, SIDE_W, SIDE_H, "Camera  /external_cam/image_raw", fit=True)
            frame[y1 + SIDE_H:y1 + 2 * SIDE_H, RV_W:] = panel(dbg, SIDE_W, SIDE_H,
                                                              "Perception  /ouija/debug_image (planche, anneau, lettre)", fit=True)
            DemoRecorder._fk_lines(frame, fk, n.tcp_from_tf(), 12, y1 + 34 + 30)
            text(frame, TITLE_REAL, (16, 30), 0.8)
            self._caption(frame, R_W, R_H, time.monotonic() - t_start, state, pointed, spelled, verdict, 1000)
            due = int((time.monotonic() - t_start) * FPS) + 1
            for _ in range(max(1, due - self.n)):
                writer.write(frame)
                self.n += 1
            time.sleep(max(0.0, t_start + self.n / FPS - time.monotonic()))
        writer.release()

    def run(self):
        if self.layout == "real":
            return self._run_real()
        import mujoco

        scene = ASSETS / (os.environ.get("SIM_SCENE") or "scene_ouija.xml")
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=MAIN_H, width=MAIN_W)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = CAM_LOOKAT
        cam.distance, cam.azimuth, cam.elevation = CAM_DISTANCE, CAM_AZIMUTH, CAM_ELEVATION
        qadr = {j: model.joint(j).qposadr[0] for j in ARM_JOINTS + ["gripper"]}
        puck = model.joint("planchette_free").qposadr[0]
        cube = model.joint("pickup_object_free").qposadr[0] if mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "pickup_object_free") >= 0 else None
        g_lo, g_hi = (float(v) for v in model.joint("gripper").range)
        u_lo, u_hi = URDF_GRIPPER_RANGE
        writer = VideoWriter(self.out, FPS, MAIN_W, MAIN_H)
        t_start = time.monotonic()
        while not self.stop.is_set():
            n = self.node
            with n.lock:
                q, gt_puck, gt_cube, dbg = dict(n.q), n.gt_puck, n.gt_cube, n.dbg
                state, pointed, spelled, verdict = n.state, n.pointed, list(n.spelled), n.verdict
            for j in ARM_JOINTS:
                if j in q:
                    data.qpos[qadr[j]] = q[j]
            if "gripper" in q:
                pct = np.clip((q["gripper"] - u_lo) / (u_hi - u_lo), 0.0, 1.0)
                data.qpos[qadr["gripper"]] = g_lo + pct * (g_hi - g_lo)
            if gt_puck is not None:
                data.qpos[puck : puck + 3] = gt_puck
                data.qpos[puck + 3 : puck + 7] = (1.0, 0.0, 0.0, 0.0)
            if cube is not None and gt_cube is not None:
                data.qpos[cube : cube + 3] = gt_cube
                data.qpos[cube + 3 : cube + 7] = (1.0, 0.0, 0.0, 0.0)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

            # inset: Ouija perception image (board, ring, letter under the ring)
            if dbg is not None:
                x0, y0 = MAIN_W - INSET_W - 16, 52
                frame[y0 : y0 + INSET_H, x0 : x0 + INSET_W] = cv2.resize(dbg, (INSET_W, INSET_H), interpolation=cv2.INTER_AREA)
                cv2.rectangle(frame, (x0, y0), (x0 + INSET_W, y0 + 28), (20, 20, 20), -1)
                text(frame, "Perception  /ouija/debug_image", (x0 + 8, y0 + 20), 0.55, (240, 240, 240), 1)
            text(frame, TITLE, (16, 36), 0.8)
            self._caption(frame, MAIN_W, MAIN_H, time.monotonic() - t_start, state, pointed, spelled, verdict, 700)
            due = int((time.monotonic() - t_start) * FPS) + 1
            for _ in range(max(1, due - self.n)):
                writer.write(frame)
                self.n += 1
            time.sleep(max(0.0, t_start + self.n / FPS - time.monotonic()))
        writer.release()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    real = "--real" in sys.argv or os.environ.get("USE_SIM", "true").lower() != "true"
    use_llm = "--ask" in sys.argv
    out = args[0] if args else ("/ros2_ws/rec/ouija_reel.mp4" if real else "/ros2_ws/rec/ouija_demo.mp4")
    question = args[1] if len(args) > 1 else "Qui est le meilleur groupe ?"
    answer = "" if use_llm else (args[2] if len(args) > 2 else "NOUS").upper()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = OuijaDemoNode()
    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    client = node.ask if use_llm else node.spell
    if not client.wait_for_service(30):
        print(f"service {'/ouija/ask' if use_llm else '/ouija/spell'} not available (is the stack in ouija mode?)")
        return
    deadline = time.time() + 30
    ready = lambda: node.q and node.dbg is not None and (real or node.gt_puck is not None)  # noqa: E731
    while time.time() < deadline and not ready():
        time.sleep(0.2)
    if not ready():
        print("no /joint_states, /ouija/debug_image" + ("" if real else " or planchette ground truth") + " yet"
              + ("" if node.dbg is not None else " (perception mute? docker compose restart perception)"))
        return

    rviz = None
    if real:
        if node.cam is None:
            print("warning: no /external_cam/image_raw yet, the camera panel will wait", flush=True)
        try:
            rviz = RvizView(RV_W, RV_H, REAL_RVIZ_CONFIG, FPS)
            print(f"rviz2 on virtual display {rviz.display}, waiting for the first frame...", flush=True)
            rviz.wait_ready()
        except Exception as exc:  # noqa: BLE001 - never lose the demo for the RViz panel
            print(f"warning: RViz panel unavailable ({exc}), recording without it", flush=True)
            if rviz is not None:
                rviz.close()
            rviz = None
    try:
        rec = Recorder(node, out, question, answer, "real" if real else "sim", rviz)
        rec.start()
        time.sleep(3.0)
        req = Ask.Request()
        req.question = question if use_llm else answer  # /ouija/spell spells the given text
        t0 = time.time()
        fut = client.call_async(req)
        while not fut.done():
            time.sleep(0.1)
        res = fut.result()
        dt = time.time() - t0
        with node.lock:
            node.verdict = (f"Reponse '{res.answer}' epelee en {dt:.0f} s" if res.success
                            else f"ECHEC : {res.message[:60]}")
        what = f"ask {question!r}" if use_llm else f"spell {answer!r}"
        print(f"{what}: success={res.success} answer={res.answer!r} in {dt:.0f} s  [{res.message[:80]}]", flush=True)
        time.sleep(4.0)
        rec.stop.set()
        rec.join()
        print(f"wrote {out}: {rec.n} frames, {rec.n / FPS:.0f} s")
    finally:
        if rviz is not None:
            rviz.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
