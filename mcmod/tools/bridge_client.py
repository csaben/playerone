"""Tiny smoke-test client for the playerone-bridge mod.

Run (Minecraft with the mod loaded, in a world):
    uv run --no-project python tools/bridge_client.py [--port 8970]

Steps: ping, hold W for 1 s, turn right for 0.5 s, release, ping.
Only the standard library is used.
"""

import argparse
import json
import socket
import threading
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8970)
    args = ap.parse_args()

    sock = socket.create_connection((args.host, args.port), timeout=3)
    sock.settimeout(None)

    def reader() -> None:
        buf = b""
        while True:
            try:
                chunk = sock.recv(65536)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                msg = json.loads(line)
                if "ok" not in msg:  # acts reply {"ok":true}; only show pongs and errors
                    print("<-", msg)

    threading.Thread(target=reader, daemon=True).start()

    def send(obj: dict) -> None:
        sock.sendall((json.dumps(obj) + "\n").encode())

    def act(keys: list[str], cam: tuple[float, float] = (0, 0), seconds: float = 0.0) -> None:
        # one act per 50 ms frame, like the sidecar; acts also feed the 1 s dead-man switch
        end = time.monotonic() + seconds
        while True:
            send({"act": {"keys": keys, "mb": [0, 0], "cam": list(cam)}})
            if time.monotonic() >= end:
                return
            time.sleep(0.05)

    print("-> ping")
    send({"ping": 1})
    time.sleep(0.3)

    print("-> hold W for 1 s")
    act(["W"], seconds=1.0)

    print("-> turn right for 0.5 s (40 counts per 50 ms)")
    act([], cam=(40, 0), seconds=0.5)

    time.sleep(0.1)  # let the last cam counts play out
    print("-> release")
    send({"release": True})
    time.sleep(0.2)

    print("-> ping")
    send({"ping": 1})
    time.sleep(0.3)
    sock.close()


if __name__ == "__main__":
    main()
