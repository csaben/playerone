"""Policy server: no-ckpt reply, reply shape with a synthetic checkpoint, hot reload flips version."""
from __future__ import annotations

import io
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
from websockets.sync.client import connect

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import dataset as D  # noqa: E402
from model import BCPolicy  # noqa: E402
from train import save_ckpt  # noqa: E402


def write_ckpt(root: Path, step: int) -> None:
    m = BCPolicy(len(D.KEY_VOCAB), len(D.MOUSE_BUTTONS), D.CAM_BINS, 4, D.IMG_SIZE)
    opt = __import__("torch").optim.AdamW(m.parameters())
    ds = SimpleNamespace(n_frames=10, chunks=[], n_takeover=0)
    save_ckpt(root / "ckpt" / "latest.pt", m, opt, SimpleNamespace(stack=4, include_agent=False), step, ds)


def jpeg(w=320, h=180, seed=0) -> bytes:
    img = np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)
    fr = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format="yuvj420p")
    c = av.CodecContext.create("mjpeg", "w")
    c.width, c.height, c.pix_fmt = w, h, "yuvj420p"
    c.time_base = __import__("fractions").Fraction(1, 20)
    pk = c.encode(fr) + c.encode(None)
    return b"".join(bytes(p) for p in pk)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_version(ws, frame, want, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        ws.send(frame)
        r = json.loads(ws.recv())
        if want(r["version"]):
            return r
        time.sleep(0.3)
    raise AssertionError(f"version never satisfied; last={r}")


def test_serve(tmp_path):
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(HERE / "serve.py"), "--host", "127.0.0.1", "--port", str(port),
                             "--data", str(tmp_path), "--poll-s", "0.5"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(100):
            try:
                ws = connect(f"ws://127.0.0.1:{port}/")
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise AssertionError("server did not start")
        with ws:
            f = jpeg()
            ws.send(f)
            r = json.loads(ws.recv())
            assert r == {"keys": [], "mb": [0, 0], "cam": [0, 0], "version": None, "infer_ms": 0.0, "step": None}

            write_ckpt(tmp_path, 1)
            r = wait_version(ws, f, lambda v: v is not None)
            assert r["version"].startswith("1-") and r["step"] == 1
            ws.send(json.dumps({"cfg": {"temperature": 0, "sample_camera": False, "key_threshold": 0.5}}))
            ws.send(json.dumps({"reset": True}))
            for i in range(5):
                ws.send(jpeg(seed=i))
                r = json.loads(ws.recv())
                assert set(r) == {"keys", "mb", "cam", "version", "infer_ms", "step"}
                assert all(k in D.KEY_VOCAB for k in r["keys"])
                assert len(r["mb"]) == 2 and all(x in (0, 1) for x in r["mb"])
                assert len(r["cam"]) == 2
                centers = np.round(D.cam_decode(np.arange(D.CAM_BINS)), 2)
                assert all(np.isclose(c, centers).any() for c in r["cam"])
            # a different size/aspect frame is fine too
            ws.send(jpeg(200, 150))
            assert "keys" in json.loads(ws.recv())

            write_ckpt(tmp_path, 2)
            r = wait_version(ws, f, lambda v: v is not None and v.startswith("2-"))
            assert r["step"] == 2
            time.sleep(2.5)
            st = json.loads((tmp_path / "status" / "serve.json").read_text())
            assert st["state"] == "serving" and st["version"].startswith("2-")
            assert len(st["clients"]) == 1 and st["clients"][0]["frames"] >= 8
    finally:
        proc.terminate()
        out = proc.communicate(timeout=10)[0]
        print(out)
