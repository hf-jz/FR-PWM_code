"""A paced loop against a control-system endpoint on loopback, with the real shield and a real socket.

This is the closest thing to hardware-in-the-loop that this machine can run without a device: the control client and
the plant-side endpoint are separate processes talking over TCP, the loop is paced at the 10 ms control period with a
wall clock, and the safety kernel that filters each command is the compiled one the paper times. What it measures is
the scheduling and interface behaviour of the closed path; what it does not measure is a device, an actuator or a
plasma, and it is reported as loopback rather than as hardware-in-the-loop.

Writes stage50/loopback_hil.json.
"""
import ast
import json
import os
import socket
import subprocess
import sys
import threading
import time

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from paths import ARTEFACTS, OUT, record  # noqa: E402
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "library")
sys.path.insert(0, REPO)

PORT = 47921
N_STEPS = 2000
PERIOD_MS = 10.0
os.makedirs("stage50", exist_ok=True)

SERVER = r'''
import json, socket, sys
port, n = int(sys.argv[1]), int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", port)); s.listen(1)
conn, _ = s.accept()
buf = b""
received = 0
while received < n:
    chunk = conn.recv(4096)
    if not chunk:
        break
    buf += chunk
    while b"\n" in buf:
        line, buf = buf.split(b"\n", 1)
        msg = json.loads(line)
        received += 1
        ack = json.dumps({"i": msg["i"], "u": msg["u"], "t_ack": __import__("time").perf_counter()})
        conn.sendall((ack + "\n").encode())
conn.close(); s.close()
print(json.dumps({"received": received}))
'''
open("stage50_server.py", "w").write(SERVER)

srv = subprocess.Popen([sys.executable, "stage50_server.py", str(PORT), str(N_STEPS)],
                       stdout=subprocess.PIPE, text=True)
time.sleep(0.6)
cli = socket.create_connection(("127.0.0.1", PORT))
cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
cli_f = cli.makefile("rwb")

from frpwm.runtime.safety_cpp import cpp_available, cbf_qp  # noqa: E402

n_state, n_act = 4, 2
G = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
h = np.array([0.4, 0.4, 0.4, 0.4])
lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])

# the pacing layer itself: what a bare sleep can hold on this host, measured before the loop
floor = []
for _ in range(30):
    t = time.perf_counter()
    time.sleep(0.005)
    floor.append((time.perf_counter() - t - 0.005) * 1e3)
floor = np.array(floor)

periods, rtts, misses, busy = [], [], 0, []
P = PERIOD_MS / 1000.0
next_t = time.perf_counter()
for i in range(N_STEPS):
    t_tick = time.perf_counter()
    u_cand = np.array([0.35 * np.sin(0.05 * i), 0.30 * np.cos(0.03 * i)])
    t0 = time.perf_counter()
    u = np.asarray(cbf_qp(u_cand, G, h, lo, hi)[0], float)
    cli_f.write((json.dumps({"i": i, "u": [float(u[0]), float(u[1])]}) + "\n").encode())
    cli_f.flush()
    ack = json.loads(cli_f.readline().decode())
    rtts.append((time.perf_counter() - t0) * 1e3)
    t_end = time.perf_counter()
    busy.append((t_end - t_tick) * 1e3)
    next_t += P
    slack = next_t - t_end
    if slack > 0.0003:
        time.sleep(slack - 0.0003)          # sleep most of the wait
        while time.perf_counter() < next_t:  # spin the last fraction, so the tick lands on time
            pass
    lateness = (time.perf_counter() - next_t) * 1e3
    if lateness > 0.5:                       # a tick counts as missed only beyond half a millisecond
        misses += 1
    periods.append((t_end - t_tick) * 1e3)

cli.close()
srv.wait(timeout=30)

per = np.array(periods)
busy = np.array(busy)
rtt = np.array(rtts)
out = dict(steps=N_STEPS, period_ms=PERIOD_MS, backend="cpp" if cpp_available() else "python",
           work_p50_ms=float(np.percentile(busy, 50)), work_p99_ms=float(np.percentile(busy, 99)),
           work_max_ms=float(busy.max()), deadline_miss_rate=float(misses / N_STEPS),
           busy_p50=float(np.percentile(busy, 50)), busy_p99=float(np.percentile(busy, 99)),
           rtt_p50=float(np.percentile(rtt, 50)), rtt_p99=float(np.percentile(rtt, 99)),
           rtt_max=float(rtt.max()),
           timer_floor_p50_ms=float(np.percentile(floor, 50)),
           timer_floor_max_ms=float(floor.max()),
           note=("loopback transport, CPU-only, no device: the loop exercises pacing, the compiled shield and the "
                 "client/server interface of the closed path, and it does not exercise an actuator or a plasma; the timer floor is the overshoot of a bare "
                 "5 ms sleep on this host, which bounds the pacing layer independently of the shield"))
json.dump(out, open("stage50/loopback_hil.json", "w"), indent=1)
print(json.dumps(out, indent=1))
