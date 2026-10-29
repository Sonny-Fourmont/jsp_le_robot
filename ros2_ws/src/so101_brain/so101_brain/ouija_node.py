#!/usr/bin/env python3
"""ouija_node.py - the robot answers questions by moving the planchette on a Ouija board.

Inputs (perception): /ouija/board_pose (board frame -> world, from the ArUco markers) and
/ouija/planchette_position (puck centre, world). The letter positions come from the
layout file (board frame) and are expressed in the world with the board pose, so the
board can be anywhere in the camera view.

Motion (brain services): /go_to_target, /move_line, /set_joint_positions.
The TCP of the arm is the tip of the fixed jaw (about 12 mm wide). The planchette is a
ring (like the window of a real planchette): the gripper is closed, the tip is lowered
into the hole and moved in a straight line to the target, dragging the ring in any
direction (no pushing geometry, no reach problem behind the ring). The ring lags behind
the tip by the play in the hole, which is compensated. After each move the arm retreats,
the camera checks where the ring is and a correction move is done if needed (closed loop).

Interfaces:
  /ouija/ask (so101_interfaces/Ask)    question -> answer (local LLM via Ollama, or rules), spelled
  /ouija/spell (so101_interfaces/Ask)  spells the given text as is (tests)
  /ouija/question (std_msgs/String)    same as /ouija/ask, fire and forget
  /ouija/answer (std_msgs/String), /ouija/state (std_msgs/String), /ouija/stop (Trigger)
State machine per character: OBSERVE -> ALIGN -> DESCEND -> MOVE -> RETREAT -> OBSERVE
(-> correction move) -> DWELL, then FIN at the end of the answer.
"""
import hashlib
import json
import math
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import rclpy
import rclpy.node
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker

from so101_interfaces.srv import Ask, GoToTarget, MoveLine, SetJointPositions

DEFAULT_LAYOUT = "/opt/so101/sim/so101_sim/assets/so101/ouija/ouija_layout.json"


# ------------------------------------------------------------------------ oracle

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower().strip())


class Oracle:
    """Scripted answers (keyword rules). Always available, used as a fallback of LlmOracle."""

    RULES = [  # (regex on the normalized question, answer)
        (r"\b(bonjour|salut|hello|coucou)\b", "SALUT"),
        (r"\b(au revoir|bye|adieu)\b", "FIN"),
        (r"\bmerci\b", "DE RIEN"),
        (r"(comment (tu )?t'?appelles|ton nom|your name|qui es.tu|who are you)", "SO101"),
        (r"(quel age|ton age|how old)", "1"),
        (r"(sens de la vie|meaning of life)", "42"),
        (r"\b(combien|how many|how much)\b", None),  # -> a digit
        (r"\b(couleur|colou?r)\b", "BLEU"),
        (r"\b(ou es.tu|ou sommes|where)\b", "EPITECH"),
        (r"\b(qui|who)\b", "AXEL"),
        (r"\b(quand|when)\b", "DEMAIN"),
    ]

    def answer(self, question: str) -> str:
        q = _normalize(question)
        digest = int(hashlib.sha1(q.encode()).hexdigest(), 16)
        for pattern, ans in self.RULES:
            if re.search(pattern, q):
                return ans if ans is not None else str(1 + digest % 9)
        # Yes/no question: a deterministic "spirit" (same question -> same answer).
        return "OUI" if digest % 2 == 0 else "NON"


def clean_answer(text: str, max_chars: int) -> str:
    """Model output -> what the board can spell: A-Z, 0-9, spaces; OUI/NON normalised."""
    text = _normalize(text).upper()
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    words = text.split()
    if not words:
        return ""
    if words[0] in ("OUI", "YES", "SI", "OK"):
        return "OUI"
    if words[0] in ("NON", "NO", "NOPE"):
        return "NON"
    text = " ".join(words)
    if len(text) > max_chars:  # cut on a word boundary when possible
        cut = text[:max_chars]
        text = cut.rsplit(" ", 1)[0] if " " in cut else cut
    return text


class LlmOracle:
    """Answers with a small local language model served by Ollama (free, offline, CPU).
    Falls back to the rule based Oracle when the server or the model is unavailable."""

    SYSTEM = (
        "Tu es l'esprit d'une planche Ouija manipulee par un bras robot. Tu reponds aux "
        "questions avec une planchette qui pointe des lettres (A-Z), des chiffres (0-9) ou "
        "les mots OUI et NON. Reponds en francais par UN SEUL mot tres court (8 lettres "
        "maximum), ou par un nombre, ou par OUI / NON. Sans accent, sans ponctuation, sans "
        "phrase, sans explication. Sois amusant et mysterieux. Exemples: 'OUI', 'NON', "
        "'42', 'PARIS', 'PIZZA', 'JAMAIS'."
    )

    def __init__(self, url, model, max_chars, timeout, logger):
        self._url = url.rstrip("/")
        self._model = model
        self._max_chars = max_chars
        self._timeout = timeout
        self._log = logger
        self._fallback = Oracle()

    def answer(self, question: str) -> str:
        try:
            raw = self._ask(question)
            ans = clean_answer(raw, self._max_chars)
            self._log.info(f"llm ({self._model}): '{raw.strip()}' -> '{ans}'")
            if ans:
                return ans
            self._log.warn("llm: empty answer, using the rules")
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, OSError) as exc:
            self._log.warn(f"llm unavailable ({exc}), using the rules")
        return self._fallback.answer(question)

    def _ask(self, question: str) -> str:
        body = json.dumps({
            "model": self._model,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 16},
            "messages": [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": question},
            ],
        }).encode()
        req = urllib.request.Request(f"{self._url}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())["message"]["content"]


# -------------------------------------------------------------------------- node

class OuijaNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("so101_ouija")
        p = self.declare_parameter
        layout_file = p("layout_file", DEFAULT_LAYOUT).value
        self._observe_joints_deg = p("observe_joints_deg", [0.0, -100.0, 40.0, -80.0, 0.0]).value
        self._safe_height = p("safe_height", 0.08).value  # TCP height for free moves (m)
        self._finger_height = p("finger_height", 0.012).value  # TCP above the board top (m); tip is 6 mm lower
        self._drag_lag = p("drag_lag", 0.0).value  # m, measured: the ring stays centred on the tip (+-2 mm)
        self._gripper_pct = p("gripper_pct", 0.0).value  # closed while moving the ring
        self._move_speed = p("move_speed", 0.05).value  # m/s on the board
        self._tolerance = p("position_tolerance", 0.008).value  # m, puck centre vs letter
        self._max_moves = p("max_moves_per_letter", 3).value
        self._dwell = p("dwell_time", 1.5).value  # s, pause on each letter
        self._detection_max_age = p("detection_max_age", 1.5).value  # s
        self._detection_timeout = p("detection_timeout", 4.0).value  # s

        layout = json.loads(Path(layout_file).read_text())
        self._targets_board = {k: np.array(v, dtype=float) for k, v in layout["targets"].items()}
        self._puck_radius = float(layout.get("planchette_radius", 0.021))
        self._puck_height = float(layout.get("planchette_height", 0.015))
        # Answer engine: "llm" = local model via Ollama (falls back to the rules), "rules".
        ai = p("ai", "llm").value
        if ai == "llm":
            self._oracle = LlmOracle(
                url=p("ollama_url", "http://ollama:11434").value,
                model=p("ollama_model", "qwen2.5:1.5b").value,
                max_chars=p("max_answer_chars", 10).value,
                timeout=p("llm_timeout", 60.0).value,
                logger=self.get_logger(),
            )
        else:
            self._oracle = Oracle()

        self._lock = threading.Lock()
        self._T_wb = None
        self._puck = None  # (xyz, monotonic time)
        self._abort = threading.Event()
        self._busy = threading.Lock()

        cb_io = MutuallyExclusiveCallbackGroup()
        cb_srv = ReentrantCallbackGroup()
        self.create_subscription(PoseStamped, "/ouija/board_pose", self._cb_board, 5, callback_group=cb_io)
        self.create_subscription(PoseStamped, "/ouija/planchette_position", self._cb_puck, 5, callback_group=cb_io)
        self.create_subscription(String, "/ouija/question", self._cb_question, 5, callback_group=cb_srv)
        self.create_subscription(PoseStamped, "/end_effector_pose", self._cb_tcp, 5, callback_group=cb_io)
        self._tcp = None
        self._pub_state = self.create_publisher(String, "/ouija/state", 5)
        self._pub_answer = self.create_publisher(String, "/ouija/answer", 5)
        self._pub_marker = self.create_publisher(Marker, "/ouija/target_marker", 5)

        self._cli_go_to = self.create_client(GoToTarget, "/go_to_target", callback_group=cb_srv)
        self._cli_line = self.create_client(MoveLine, "/move_line", callback_group=cb_srv)
        self._cli_joints = self.create_client(SetJointPositions, "/set_joint_positions", callback_group=cb_srv)
        self.create_service(Ask, "/ouija/ask", self._srv_ask, callback_group=cb_srv)
        self.create_service(Ask, "/ouija/spell", self._srv_spell, callback_group=cb_srv)
        self.create_service(Trigger, "/ouija/stop", self._srv_stop, callback_group=cb_srv)
        self._set_state("IDLE")
        self.get_logger().info(f"Ouija node ready ({len(self._targets_board)} targets).")

    # ------------------------------------------------------------------ inputs

    def _cb_board(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, [p.x, p.y, p.z]
        with self._lock:
            self._T_wb = T

    def _cb_tcp(self, msg: PoseStamped):
        p = msg.pose.position
        with self._lock:
            self._tcp = np.array([p.x, p.y, p.z])

    def _cb_puck(self, msg: PoseStamped):
        p = msg.pose.position
        with self._lock:
            self._puck = (np.array([p.x, p.y, p.z]), time.monotonic())

    def _cb_question(self, msg: String):
        if not self._busy.acquire(blocking=False):
            self.get_logger().warn("busy, question ignored")
            return
        try:
            self._answer(msg.data)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"failed: {exc}")
        finally:
            self._set_state("IDLE")
            self._busy.release()

    def _target_world(self, label):
        with self._lock:
            T = self._T_wb
        if T is None:
            raise RuntimeError("board pose unknown (no /ouija/board_pose): are the ArUco markers visible?")
        bx, by = self._targets_board[label]
        return (T @ np.array([bx, by, 0.0, 1.0]))[:3]

    def _fresh_puck(self):
        with self._lock:
            if self._puck is None or time.monotonic() - self._puck[1] > self._detection_max_age:
                return None
            return self._puck[0].copy()

    # --------------------------------------------------------------- helpers

    def _set_state(self, state):
        self._state = state
        self._pub_state.publish(String(data=state))
        self.get_logger().info(f"[{state}]")

    def _check_abort(self):
        if self._abort.is_set():
            raise RuntimeError("aborted by /ouija/stop")

    def _call(self, client, req, timeout=60.0):
        self._check_abort()
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"service {client.srv_name} not available")
        fut = client.call_async(req)
        done = threading.Event()
        fut.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            raise RuntimeError(f"{client.srv_name}: timeout")
        return fut.result()

    @staticmethod
    def _pose(position, jaw_dir=None):
        """PoseStamped in the world; jaw_dir (xy) -> orientation with z down and x = jaw axis."""
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, position)
        if jaw_dir is None:
            msg.pose.orientation.w = 1.0  # default: from above, roll free
        else:
            x = np.array([jaw_dir[0], jaw_dir[1], 0.0])
            x /= np.linalg.norm(x)
            z = np.array([0.0, 0.0, -1.0])
            y = np.cross(z, x)
            R = np.column_stack([x, y, z])
            # rotation matrix -> quaternion (w > 0)
            w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
            if w > 1e-6:
                qx, qy, qz = (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)
            else:  # 180 deg rotation (z down, x = +x world): axis in the xy plane
                qx, qy, qz = math.sqrt(max(0.0, (1 + R[0, 0]) / 2)), math.sqrt(max(0.0, (1 + R[1, 1]) / 2)), 0.0
                if R[0, 1] < 0:
                    qy = -qy
            msg.pose.orientation.x, msg.pose.orientation.y = float(qx), float(qy)
            msg.pose.orientation.z, msg.pose.orientation.w = float(qz), float(w)
        return msg

    def _go_to(self, position, jaw_dir=None, what="go_to", tolerant=False):
        req = GoToTarget.Request()
        req.target_pose = self._pose(position, jaw_dir)
        res = self._call(self._cli_go_to, req)
        if not res.success:
            if tolerant and res.message.startswith("reached"):  # a bit off, the next line fixes it
                self.get_logger().warn(f"{what}: {res.message}")
                return
            raise RuntimeError(f"{what}: {res.message}")

    def _line(self, position, jaw_dir=None, speed=None, what="move_line"):
        req = MoveLine.Request()
        req.target_pose = self._pose(position, jaw_dir)
        req.speed = float(speed if speed is not None else self._move_speed)
        res = self._call(self._cli_line, req)
        if not res.success:
            raise RuntimeError(f"{what}: {res.message}")

    def _line_precise(self, position, jaw_dir, speed, what, tolerance=0.003, tries=3):
        """move_line, repeated until the measured TCP (FK of /joint_states) is within
        tolerance of the target: compensates the sag of the arm before a push."""
        for _ in range(tries):
            self._line(position, jaw_dir, speed, what)
            time.sleep(0.2)
            with self._lock:
                tcp = None if self._tcp is None else self._tcp.copy()
            if tcp is None:
                return
            err = float(np.linalg.norm(tcp[:2] - np.asarray(position[:2])))
            if err <= tolerance:
                return
            self.get_logger().info(f"{what}: TCP {err * 1000:.1f} mm off, correcting")

    def _joints(self, joints_deg, gripper_pct=None):
        req = SetJointPositions.Request()
        req.positions = [float(v) for v in joints_deg] + ([] if gripper_pct is None else [float(gripper_pct)])
        res = self._call(self._cli_joints, req)
        if not res.success:
            raise RuntimeError(f"set_joint_positions: {res.message}")

    def _publish_target_marker(self, P, label):
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "world"
        m.ns, m.id, m.type, m.action = "ouija_target", 0, Marker.CYLINDER, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(P[0]), float(P[1]), float(P[2]) + 0.001
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = 2 * self._puck_radius + 0.01
        m.scale.z = 0.002
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.2, 0.7
        m.text = label
        self._pub_marker.publish(m)

    # ------------------------------------------------------------ primitives

    def _observe(self):
        """Arm out of the camera view, then wait for a fresh planchette detection."""
        self._set_state("OBSERVE")
        self._joints(self._observe_joints_deg, self._gripper_pct)
        t_end = time.monotonic() + self._detection_timeout
        while time.monotonic() < t_end:
            self._check_abort()
            puck = self._fresh_puck()
            if puck is not None:
                return puck
            time.sleep(0.1)
        raise RuntimeError("planchette not detected (hidden or off the board?)")

    def _move_puck_to(self, target, label):
        """Closed loop: observe, drag the ring, observe again, correct. Returns the final error."""
        self._publish_target_marker(target, label)
        err = None
        for attempt in range(self._max_moves + 1):
            puck = self._observe()
            d = target[:2] - puck[:2]
            dist = float(np.linalg.norm(d))
            self.get_logger().info(f"{label}: planchette at ({puck[0]:.3f}, {puck[1]:.3f}), {dist * 1000:.0f} mm away")
            err = dist
            if dist <= self._tolerance or attempt == self._max_moves:
                break
            u = d / dist
            z = float(puck[2]) - self._puck_height / 2 + self._finger_height  # ring centre -> board top
            end = target[:2] + u * self._drag_lag  # the ring stays a little behind the tip
            self._set_state("ALIGN")
            self._go_to([puck[0], puck[1], self._safe_height], what="align above the planchette", tolerant=True)
            self._set_state("DESCEND")
            self._line_precise([puck[0], puck[1], self._safe_height], None, 0.08, "centre above the planchette")
            self._line([puck[0], puck[1], z], None, speed=0.04, what="finger into the planchette")
            self._set_state("MOVE")
            self._line([*end, z], None, what=f"move to {label}")
            time.sleep(0.2)
            self._set_state("RETREAT")
            self._line([*end, self._safe_height], None, speed=0.08, what="retreat")
        return err

    def _spell(self, text):
        """Moves the planchette on each character of text, then on FIN."""
        words = {"OUI", "NON", "FIN"}
        text = _normalize(text).upper()
        sequence = [text] if text in words else [c for c in text if c in self._targets_board]
        if not sequence:
            raise RuntimeError(f"nothing to spell in '{text}'")
        previous = None
        for label in sequence + ["FIN"]:
            self._check_abort()
            target = self._target_world(label)
            if label == previous:  # same letter twice: step aside first so that it shows
                side = target + np.array([0.0, 0.035, 0.0]) if target[1] < 0 else target - np.array([0.0, 0.035, 0.0])
                self._move_puck_to(side, f"{label} (again)")
            err = self._move_puck_to(target, label)
            self._set_state(f"DWELL {label} ({err * 1000:.0f} mm)")
            time.sleep(self._dwell)
            previous = label
        self._observe()

    def _answer(self, question):
        answer = self._oracle.answer(question)
        self.get_logger().info(f"question: '{question}' -> answer: '{answer}'")
        self._pub_answer.publish(String(data=answer))
        self._spell(answer)
        return answer

    # --------------------------------------------------------------- services

    def _run(self, res, fn, *args):
        if not self._busy.acquire(blocking=False):
            res.success, res.message = False, "busy (a question is being answered)"
            return res
        self._abort.clear()
        try:
            res.answer = fn(*args)
            res.success, res.message = True, "done"
        except Exception as exc:  # noqa: BLE001
            res.success, res.message = False, str(exc)
            self.get_logger().error(f"failed: {exc}")
        finally:
            self._set_state("IDLE")
            self._busy.release()
        return res

    def _srv_ask(self, req: Ask.Request, res: Ask.Response):
        return self._run(res, self._answer, req.question)

    def _srv_spell(self, req: Ask.Request, res: Ask.Response):
        def spell(text):
            self._spell(text)
            return text
        return self._run(res, spell, req.question)

    def _srv_stop(self, req, res: Trigger.Response):
        self._abort.set()
        res.success, res.message = True, "stop requested"
        return res


def main(args=None):
    rclpy.init(args=args)
    node = OuijaNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
