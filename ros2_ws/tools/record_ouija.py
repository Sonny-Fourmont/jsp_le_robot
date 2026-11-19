#!/usr/bin/env python3
"""record_ouija.py - records a video of the Ouija demo in simulation, without a screen.

3D view of the simulation (MuJoCo replay of /joint_states and of the planchette's ground truth,
rendered off-screen, 1280x720) with the Ouija perception image as an inset (/ouija/debug_image:
board, ring, letter under the ring) and a caption: question, letters spelled so far (/ouija/state),
letter seen under the ring by the perception (/ouija/pointed). The answer is spelled with
/ouija/spell (the word is given, no LLM), so the video is predictable.
Usage: record_ouija.py [out.mp4] ["question"] [ANSWER]

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_demo import ARM_JOINTS, ASSETS, FPS, MAIN_H, MAIN_W, URDF_GRIPPER_RANGE, VideoWriter, img_from_msg, text  # noqa: E402

TITLE = "SO-ARM101  planche Ouija en simulation (MuJoCo + ROS 2)"
INSET_W, INSET_H = 400, 300
# over the robot's shoulder: the letters of the board are printed to be read from the robot's side
CAM_LOOKAT, CAM_DISTANCE, CAM_AZIMUTH, CAM_ELEVATION = (0.25, 0.03, 0.0), 0.9, 38.0, -52.0


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
        self.dbg = None
        self.state, self.pointed, self.answer = "IDLE", "", ""
        self.spelled = []  # one entry per DWELL on a letter
        self.verdict = ""
        best_effort = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, "/joint_states", self._cb_js, 10)
        self.create_subscription(PoseStamped, "/debug/planchette_position_gt", lambda m: self._set_pos("gt_puck", m), 5)
        self.create_subscription(PoseStamped, "/debug/object_position_gt", lambda m: self._set_pos("gt_cube", m), 5)
        self.create_subscription(Image, "/ouija/debug_image", self._cb_dbg, best_effort)
        self.create_subscription(String, "/ouija/state", self._cb_state, 5)
        self.create_subscription(String, "/ouija/pointed", self._cb_pointed, 5)
        self.create_subscription(String, "/ouija/answer", self._cb_answer, 5)
        self.spell = self.create_client(Ask, "/ouija/spell")

    def _cb_js(self, msg):
        with self.lock:
            self.q = dict(zip(msg.name, msg.position))

    def _set_pos(self, key, msg):
        p = msg.pose.position
        with self.lock:
            setattr(self, key, np.array([p.x, p.y, p.z]))

    def _cb_dbg(self, msg):
        img = img_from_msg(msg)
        with self.lock:
            self.dbg = img

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
    def __init__(self, node, out, question, answer):
        super().__init__(daemon=True)
        self.node, self.out, self.question, self.answer = node, out, question, answer
        self.stop = threading.Event()
        self.n = 0

    def run(self):
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
        question = strip_accents(self.question)
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
            # caption bar: state, question, letters spelled so far, letter seen under the ring
            bar = 126
            cv2.rectangle(frame, (0, MAIN_H - bar), (MAIN_W, MAIN_H), (20, 20, 20), -1)
            t = time.monotonic() - t_start
            text(frame, f"Etat : {state}   {t:5.1f} s", (16, MAIN_H - bar + 34), 0.8, (255, 220, 120))
            shown = " ".join(spelled) + (" _" * max(0, len(self.answer) - len(spelled)) if not verdict else "")
            text(frame, f"Question : {question}", (16, MAIN_H - bar + 68), 0.7, (255, 255, 255))
            text(frame, f"Reponse epelee : {shown.strip()}", (16, MAIN_H - bar + 102), 0.8, (120, 255, 150))
            if pointed:
                text(frame, f"Lettre sous l'anneau (perception) : {pointed}", (700, MAIN_H - bar + 68), 0.7, (160, 230, 255))
            if verdict:
                text(frame, verdict, (700, MAIN_H - bar + 102), 0.7, (120, 255, 150))
            due = int((time.monotonic() - t_start) * FPS) + 1
            for _ in range(max(1, due - self.n)):
                writer.write(frame)
                self.n += 1
            time.sleep(max(0.0, t_start + self.n / FPS - time.monotonic()))
        writer.release()


def main():
    args = sys.argv[1:]
    out = args[0] if args else "/ros2_ws/rec/ouija_demo.mp4"
    question = args[1] if len(args) > 1 else "Qui est le meilleur groupe ?"
    answer = (args[2] if len(args) > 2 else "NOUS").upper()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = OuijaDemoNode()
    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    if not node.spell.wait_for_service(30):
        print("service /ouija/spell not available (is the stack in ouija mode?)")
        return
    deadline = time.time() + 30
    while time.time() < deadline and not (node.q and node.gt_puck is not None and node.dbg is not None):
        time.sleep(0.2)
    if not node.q or node.gt_puck is None or node.dbg is None:
        print("no /joint_states, planchette ground truth or /ouija/debug_image yet")
        return

    rec = Recorder(node, out, question, answer)
    rec.start()
    time.sleep(3.0)
    req = Ask.Request()
    req.question = answer  # /ouija/spell spells the given text
    t0 = time.time()
    fut = node.spell.call_async(req)
    while not fut.done():
        time.sleep(0.1)
    res = fut.result()
    dt = time.time() - t0
    with node.lock:
        node.verdict = (f"Reponse '{res.answer}' epelee en {dt:.0f} s" if res.success
                        else f"ECHEC : {res.message[:60]}")
    print(f"spell {answer!r}: success={res.success} answer={res.answer!r} in {dt:.0f} s  [{res.message[:80]}]", flush=True)
    time.sleep(4.0)
    rec.stop.set()
    rec.join()
    print(f"wrote {out}: {rec.n} frames, {rec.n / FPS:.0f} s")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
