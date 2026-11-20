"""grasp_bench.py - offline pick & place bench: MuJoCo physics + the brain's own IK, no ROS, ~5 s per trial.

Replays the brain's sequence (observe pose, transit, vertical pre-grasp, straight descend, grasp,
straight lift, transport to the box, release) over a grid of cube positions with perfect perception,
and reports the cube's final position (in the box or not). Use it to tune grasp_offset / grasp_height
or to check a change of the grasp logic before running the ROS stack.

    docker compose exec control bash -lc "source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash \
        && python3 /ros2_ws/tools/grasp_bench.py -7 30 4"          # offset mm, TCP height mm, extra random cubes
    docker compose exec -e OBJ_ERR=8,0 control ...                   # add a perception error (mm) to every cube
    ... grasp_bench.py -7 30 0 '0.19,0.02;0.26,-0.1'                 # explicit cube positions (m)
env IMPRATIO / NOSLIP override the scene's MuJoCo options (to compare friction settings)."""
import importlib.util, math, os, sys, threading, types
import numpy as np, pinocchio as pin, mujoco
args = sys.argv[1:]; sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("brain_new", "/ros2_ws/src/so101_brain/so101_brain/brain_node.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
class Fake(mod.BrainNode):
    def __init__(self):
        self._model = pin.buildModelFromUrdf(mod._urdf_path()); self._data = self._model.createData(); self._tcp_id = self._model.getFrameId(mod.TCP_FRAME)
        self._q_lo = np.array(self._model.lowerPositionLimit[:5]) + mod.IK_LIMIT_MARGIN; self._q_hi = np.array(self._model.upperPositionLimit[:5]) - mod.IK_LIMIT_MARGIN
        self._q = np.concatenate([np.radians([0, -80, 80, 60, 0]), [1.7]]); self._lock = threading.Lock()
    def get_logger(self): return types.SimpleNamespace(warn=lambda *a: None, warning=lambda *a: None, info=lambda *a: None)
S = Fake()
model = mujoco.MjModel.from_xml_path("/opt/so101/sim/so101_sim/assets/so101/scene_tek5.xml"); data = mujoco.MjData(model)
if os.environ.get("IMPRATIO"): model.opt.impratio = float(os.environ["IMPRATIO"])
if os.environ.get("NOSLIP"): model.opt.noslip_iterations = int(os.environ["NOSLIP"])
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
qadr = {j: model.joint(j).qposadr[0] for j in ARM + ["gripper"]}; aid = {j: model.actuator(j).id for j in ARM + ["gripper"]}
cube = model.joint("pickup_object_free").qposadr[0]; box = model.body("drop_box").pos.copy()
G_OPEN, G_CLOSED = 1.745, -0.1745; dt = model.opt.timestep
d = float(args[0]) / 1000 if len(args) > 0 else 0.028; h = float(args[1]) / 1000 if len(args) > 1 else 0.012; n_rand = int(args[2]) if len(args) > 2 else 0
print(f"d={d*1000:.0f} mm h={h*1000:.0f} mm impratio={model.opt.impratio} noslip={model.opt.noslip_iterations} box={np.round(box[:2],3)}")
OBSERVE = np.radians([0, -90, -5, 90, 0])
class Arm:
    def __init__(self): self.q = OBSERVE.copy()
    def ik(self, p, local, x_dir=None, symmetric=True):
        S._q = np.concatenate([self.q, [G_OPEN]])
        q, err, _ = S._solve_ik(np.array(p), x_dir=x_dir, local=local, symmetric=symmetric)
        if local and (q is None or err > 0.01): q, err, _ = S._solve_ik(np.array(p), x_dir=x_dir, symmetric=symmetric)
        if q is None or err > 0.01: raise RuntimeError(f"IK {np.round(p,3)}: {err*1000:.0f} mm")
        return q
    def go(self, p, local, jaw_offset, straight=False, grip=G_OPEN, speed=1.0):
        """brain go_to: multi-start or local IK, then jaw offset along the solution's jaw axis, optional straight path."""
        x_dir, sym = (S._jaw_axis(self.q), False) if local else (None, True)
        q = self.ik(p, local, x_dir, sym)
        target = np.array(p, float)
        if jaw_offset:
            jaw = S._jaw_axis(q); target = target + jaw_offset * jaw
            S._q = np.concatenate([q, [G_OPEN]])
            q2, err, _ = S._solve_ik(target, q_seed=np.concatenate([q, [G_OPEN]]), x_dir=jaw, local=True, symmetric=False)
            if q2 is None or err > 0.01: q2, err, _ = S._solve_ik(target, x_dir=jaw, symmetric=False)
            if q2 is None or err > 0.01: raise RuntimeError(f"IK grasp point: {err*1000:.0f} mm")
            q = q2
        path = [q]
        if straight:
            p0 = S._forward_kinematics(np.concatenate([self.q, [G_OPEN]])).translation; n = int(math.ceil(np.linalg.norm(target - p0) / 0.01)); jaw = S._jaw_axis(q); path = []
            seed = self.q
            for i in range(1, n):
                qi, err, _ = S._solve_ik(p0 + (target - p0) * i / n, q_seed=np.concatenate([seed, [G_OPEN]]), x_dir=jaw, local=True, symmetric=False)
                if qi is None or err > 0.01: path = []; break
                path.append(qi); seed = qi
            path.append(q)
        for q_to in path:
            dur = max(0.3, float(np.max(np.abs(q_to - self.q))) / speed) if not straight else 0.01 / 0.12
            self.run(dur, self.q, q_to, grip); self.q = q_to
        self.run(0.3, self.q, self.q, grip)  # MOTION_HOLD
        return target
    def run(self, seconds, q_from, q_to, grip):
        n = max(1, int(seconds / dt))
        for k in range(n):
            q = q_from + (q_to - q_from) * min(1.0, (k + 1) / n)
            for j, v in zip(ARM, q): data.ctrl[aid[j]] = v
            data.ctrl[aid["gripper"]] = grip; mujoco.mj_step(model, data)
            self.zmin = min(getattr(self, 'zmin', 9.0), float(data.site('gripperframe').xpos[2]))
def grip_pct(): return (data.qpos[qadr["gripper"]] - G_CLOSED) / (G_OPEN - G_CLOSED) * 100
def trial(cx, cy):
    mujoco.mj_resetData(model, data); arm = Arm()
    for j, v in zip(ARM, OBSERVE): data.qpos[qadr[j]] = v
    data.qpos[qadr["gripper"]] = G_OPEN; data.qpos[cube:cube + 3] = (cx, cy, 0.025); data.qpos[cube + 3:cube + 7] = (1, 0, 0, 0)
    mujoco.mj_forward(model, data); arm.run(0.3, OBSERVE, OBSERVE, G_OPEN)
    obj = data.qpos[cube:cube + 3].copy()  # perfect perception
    if os.environ.get('OBJ_ERR'): obj[:2] += np.array([float(v) for v in os.environ['OBJ_ERR'].split(',')]) / 1000  # simulated perception error (mm)
    try:
        arm.zmin = 9.0; arm.go([obj[0], obj[1], 0.12], local=False, jaw_offset=d); dip = arm.zmin; cube_moved = np.linalg.norm(data.qpos[cube:cube+2] - obj[:2]) * 1000
        z_app = 0.12
        for z in (0.12, 0.10, 0.08, 0.06):
            S._q = np.concatenate([arm.q, [G_OPEN]]); q_try, e_try, a_try = S._solve_ik(np.array([obj[0], obj[1], z]))
            if q_try is not None and e_try <= 0.01 and a_try < mod.IK_AXIS_EPS:
                if z < 0.12: arm.go([obj[0], obj[1], z], local=True, jaw_offset=d, straight=True)
                z_app = z; break
        arm.go([obj[0], obj[1], h], local=True, jaw_offset=d, straight=True)
        Rg = data.site("gripperframe").xmat.reshape(3, 3); tilt = math.degrees(math.acos(min(1.0, abs(Rg[2, 0]))))  # site x = approach axis (down)
        extra = f"approach z={z_app*100:.0f} cm (TCP dipped to {dip*100:.1f} cm, cube moved {cube_moved:.0f} mm), gripper tilt {tilt:.0f} deg, tcp z={data.site('gripperframe').xpos[2]*1000:.0f} mm"
        arm.run(0.8, arm.q, arm.q, G_CLOSED); g1 = grip_pct(); tcp_cube = data.site("gripperframe").xpos - data.qpos[cube:cube + 3]
        arm.go([obj[0], obj[1], 0.15], local=True, jaw_offset=d, straight=True, grip=G_CLOSED); z_lift = data.qpos[cube + 2]
        arm.go([box[0], box[1], 0.15], local=True, jaw_offset=d, grip=G_CLOSED)
        arm.go([box[0], box[1], 0.10], local=True, jaw_offset=d, grip=G_CLOSED); z_before = data.qpos[cube + 2]
        arm.run(0.8, arm.q, arm.q, G_OPEN); arm.go([box[0], box[1], 0.15], local=True, jaw_offset=d)
        arm.run(0.5, arm.q, arm.q, G_OPEN)
    except RuntimeError as e:
        return False, f"{e}"
    c = data.qpos[cube:cube + 3]; in_box = np.linalg.norm(c[:2] - box[:2]) < 0.05 and c[2] < 0.06
    return in_box, f"{extra} | grip {g1:3.0f} % tcp-cube ({tcp_cube[0]*1000:+3.0f},{tcp_cube[1]*1000:+3.0f}) lift z={z_lift*1000:3.0f} transport z={z_before*1000:3.0f} end ({c[0]:.2f},{c[1]:.2f},{c[2]*1000:.0f})"
rng = np.random.default_rng(1)
cases = [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a))) for r in (0.18, 0.22, 0.26, 0.30) for a in (-50, -25, 0, 25, 50)]
if len(args) > 3: cases = [tuple(float(v) for v in c.split(",")) for c in args[3].split(";")]
cases += [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a))) for r, a in zip(rng.uniform(0.18, 0.30, n_rand), rng.uniform(-50, 60, n_rand))]
ok = 0
for cx, cy in cases:
    res, msg = trial(cx, cy); ok += res
    print(f"cube ({cx:+.3f},{cy:+.3f}) r={math.hypot(cx,cy):.2f} a={math.degrees(math.atan2(cy,cx)):+3.0f}: {'OK  ' if res else 'FAIL'} {msg}", flush=True)
print(f"success {ok}/{len(cases)}")
