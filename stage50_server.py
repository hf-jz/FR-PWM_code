
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
