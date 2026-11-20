"""perception_bias.py - simulation only: perception error map of the cube detection.

Puts the cube at N random positions (/sim/randomize_object) with the arm in the observe pose and prints
/object_position minus the ground truth for each one (mm), then the mean bias and the worst error.

    docker compose exec control bash -lc "source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash \
        && python3 /ros2_ws/tools/perception_bias.py 30"
"""
import math, sys, threading, time, numpy as np, rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from so101_interfaces.srv import SetJointPositions
rclpy.init(); n = Node("bias_probe"); st = {}
def pos(m): return np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])
n.create_subscription(PoseStamped, "/object_position", lambda m: st.__setitem__("det", (time.time(), pos(m))), 5)
n.create_subscription(PoseStamped, "/debug/object_position_gt", lambda m: st.__setitem__("gt", pos(m)), 5)
threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()
sj = n.create_client(SetJointPositions, "/set_joint_positions"); rnd = n.create_client(Trigger, "/sim/randomize_object")
sj.wait_for_service(10); rnd.wait_for_service(10)
req = SetJointPositions.Request(); req.positions = [0.0, -90.0, -5.0, 90.0, 0.0, 100.0]  # brain observe pose
f = sj.call_async(req); t0 = time.time()
while not f.done() and time.time() - t0 < 10: time.sleep(0.05)
time.sleep(1.0)
rows = []
for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 30):
    f = rnd.call_async(Trigger.Request()); t0 = time.time()
    while not f.done() and time.time() - t0 < 5: time.sleep(0.05)
    time.sleep(0.8)
    if "det" not in st or time.time() - st["det"][0] > 0.5:
        print(f"{i}: gt=({st['gt'][0]:.3f},{st['gt'][1]:.3f}) NOT DETECTED"); continue
    e = (st["det"][1] - st["gt"])[:2] * 1000; gt = st["gt"]
    r, a = math.hypot(gt[0], gt[1]), math.degrees(math.atan2(gt[1], gt[0]))
    rows.append((r, a, e[0], e[1]))
    print(f"{i:2d}: gt=({gt[0]:.3f},{gt[1]:.3f}) r={r:.2f} a={a:+4.0f}  det-gt=({e[0]:+5.1f},{e[1]:+5.1f}) mm  |e|={math.hypot(*e):4.1f}")
E = np.array(rows)
print(f"n={len(E)} mean bias ({E[:,2].mean():+.1f},{E[:,3].mean():+.1f}) mm, |e| mean {np.hypot(E[:,2],E[:,3]).mean():.1f} max {np.hypot(E[:,2],E[:,3]).max():.1f} mm")
rclpy.shutdown()
