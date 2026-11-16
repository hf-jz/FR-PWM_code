"""真实 PCS 对接（Stage C 前置）：TCP 帧协议 + watchdog，跨进程/跨机可部署。
协议（length-prefixed JSON，4 字节大端长度头 + UTF-8 JSON）：
  客户端 → {"type":"u_safe","frame_id":n,"u":{"heat":x,...},"prov":{...}} → 服务端回 {"accepted":bool,...}
  客户端 → {"type":"heartbeat"} → 服务端回 {"ok":true}
  服务端 watchdog：连续 heartbeat_miss_max 个周期无心跳 → 联锁，拒收 u_safe 帧（回退传统控制语义）。
执行器 slew/range 钳制与帧序号检查由 PcsEmulator 内核保证（附录 C）。"""
import json, socket, struct, threading, time
from .pcs import PcsEmulator

def _send(sock, obj):
    b = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack(">I", len(b)) + b)

def _recv(sock):
    hdr = sock.recv(4)
    if len(hdr) < 4:
        return None
    (n,) = struct.unpack(">I", hdr)
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return json.loads(buf.decode("utf-8"))

class PcsLinkServer:
    def __init__(self, host="127.0.0.1", port=0, heartbeat_period=0.005, miss_max=3):
        self.core = PcsEmulator(heartbeat_miss_max=miss_max)
        self.period, self.miss_max = heartbeat_period, miss_max
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self._stop = threading.Event()
        self._th = None
    def start(self):
        self._th = threading.Thread(target=self._serve, daemon=True)
        self._th.start()
        return self.port
    def _serve(self):
        conn, _ = self.sock.accept()
        conn.settimeout(self.period)
        last_hb = time.time()
        self.core.last_hb = last_hb          # watchdog 窗口自链路建立起算
        self.core.interlock = False
        while not self._stop.is_set():
            try:
                msg = _recv(conn)
            except socket.timeout:
                msg = None
            if time.time() - last_hb > self.period * self.miss_max:
                self.core.interlock = True
            if msg is None:
                continue
            t = msg.get("type")
            if t == "heartbeat":
                last_hb = time.time(); self.core.heartbeat()
                _send(conn, {"ok": True})
            elif t == "u_safe":
                if self.core.interlock:
                    _send(conn, {"accepted": False, "reason": "INTERLOCK"})
                    continue
                u, ok = self.core.apply(msg.get("u", {}), frame_id=msg.get("frame_id"),
                                        provenance=msg.get("prov"))
                _send(conn, {"accepted": ok, "u": u,
                             "reason": None if ok else "SLEW_OR_STALE_FRAME"})
    def stop(self):
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass

class PcsLinkClient:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=2.0)
    def heartbeat(self):
        _send(self.sock, {"type": "heartbeat"})
        return _recv(self.sock)["ok"]
    def send_u(self, u, frame_id, prov=None):
        _send(self.sock, {"type": "u_safe", "frame_id": frame_id, "u": u, "prov": prov or {}})
        return _recv(self.sock)
    def close(self):
        self.sock.close()

if __name__ == "__main__":
    srv = PcsLinkServer()
    port = srv.start()
    time.sleep(0.05)
    cli = PcsLinkClient("127.0.0.1", port)
    r1 = cli.send_u({"heat": 0.9}, 1)                      # slew 钳制到 0.2
    assert r1["accepted"] and abs(r1["u"]["heat"] - 0.2) < 1e-9, r1
    r2 = cli.send_u({"heat": 0.0}, 1)                      # 重复帧号拒绝
    assert not r2["accepted"] and r2["reason"] == "SLEW_OR_STALE_FRAME"
    cli.heartbeat()
    r3 = cli.send_u({"heat": 0.0}, 2)
    assert r3["accepted"]
    time.sleep(0.05)                                       # 心跳中断 → 联锁
    r4 = cli.send_u({"heat": 0.5}, 3)
    assert not r4["accepted"] and r4["reason"] == "INTERLOCK"
    cli.heartbeat()                                        # 心跳恢复 → 解除联锁
    time.sleep(0.01)
    r5 = cli.send_u({"heat": 0.1}, 4)
    assert r5["accepted"]
    cli.close(); srv.stop()
    print(f"pcs_link OK: slew/帧序号/联锁/恢复 全链路(端口 {port})")
