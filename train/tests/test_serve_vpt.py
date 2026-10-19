"""VPT backend over the same WS contract. Skipped unless the VPT repo + weights are present (cradle)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from websockets.sync.client import connect

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

from test_serve import free_port, jpeg  # noqa: E402

DATA = Path(os.environ.get("DATA", "~/player-one-data")).expanduser()
WEIGHTS = "rl-from-early-game-2x.weights"
REPO = Path(os.environ.get("VPT_REPO", "~/Code/player-one/third_party/Video-Pre-Training")).expanduser()
pytestmark = pytest.mark.skipif(not ((DATA / "vpt" / WEIGHTS).exists() and REPO.exists()),
                                reason="VPT repo/weights not present")


def test_to_16x9():
    import numpy as np
    from vpt_backend import to_16x9
    tall, wide = np.ones((480, 640, 3), np.uint8), np.ones((200, 600, 3), np.uint8)
    assert to_16x9(tall, "crop").shape == (360, 640, 3)
    assert to_16x9(wide, "crop").shape == (200, 356, 3)
    assert to_16x9(tall, "pad").shape == (480, 853, 3) and to_16x9(wide, "pad").shape == (338, 600, 3)
    assert to_16x9(tall, "stretch").shape == tall.shape
    assert to_16x9(np.ones((180, 320, 3), np.uint8), "crop").shape == (180, 320, 3)


def test_serve_vpt(tmp_path):
    (tmp_path / "vpt").mkdir()
    for f in ("2x.model", WEIGHTS):
        (tmp_path / "vpt" / f).symlink_to(DATA / "vpt" / f)
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(HERE / "serve.py"), "--policy", "vpt", "--host", "127.0.0.1",
                             "--port", str(port), "--data", str(tmp_path)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(300):  # model load takes a few seconds
            try:
                ws = connect(f"ws://127.0.0.1:{port}/")
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise AssertionError("server did not start")
        with ws:
            ws.send(json.dumps({"reset": True}))
            seen_keys = set()
            for i in range(30):
                ws.send(jpeg(320, 180, seed=i))
                r = json.loads(ws.recv())
                assert set(r) == {"keys", "mb", "cam", "version", "infer_ms", "step"}, r
                assert r["version"] == "vpt:rl-from-early-game-2x"
                assert "ESC" not in r["keys"] and len(r["mb"]) == 2 and len(r["cam"]) == 2
                assert all(abs(c) <= 10 / 0.15 + 0.01 for c in r["cam"]), r  # VPT max 10 deg/frame
                seen_keys.update(r["keys"])
            ws.send(json.dumps({"reset": True, "cfg": {"sample_camera": False}}))
            ws.send(jpeg())
            assert "keys" in json.loads(ws.recv())
            print("keys seen:", seen_keys, "last:", r)
        time.sleep(2.5)
        st = json.loads((tmp_path / "status" / "serve-vpt.json").read_text())
        assert st["policy"] == "vpt" and st["version"] == "vpt:rl-from-early-game-2x"
    finally:
        proc.terminate()
        print(proc.communicate(timeout=10)[0][-3000:])


def test_serve_vptft_hot_reload(tmp_path):
    (tmp_path / "vpt").mkdir()
    for f in ("2x.model", WEIGHTS):
        (tmp_path / "vpt" / f).symlink_to(DATA / "vpt" / f)
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(HERE / "serve.py"), "--policy", "vpt", "--vpt-ft", "--host",
                             "127.0.0.1", "--port", str(port), "--data", str(tmp_path), "--poll-s", "0.5"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(300):
            try:
                ws = connect(f"ws://127.0.0.1:{port}/")
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise AssertionError("server did not start")
        with ws:
            ws.send(jpeg())
            assert "no fine-tune yet" in json.loads(ws.recv())["version"]
            ck = tmp_path / "vpt-ckpt"
            ck.mkdir()
            (ck / "latest.weights").symlink_to(DATA / "vpt" / WEIGHTS)  # a policy state_dict, like vpt_train saves
            (ck / "latest.json").write_text(json.dumps({"version": "vptft:7-123", "step": 7, "path": "latest.weights",
                                                        "n_frames": 1}))
            for _ in range(60):
                ws.send(jpeg())
                r = json.loads(ws.recv())
                if r["version"] == "vptft:7-123":
                    break
                time.sleep(0.3)
            assert r["version"] == "vptft:7-123" and r["step"] == 7
        time.sleep(2.5)
        st = json.loads((tmp_path / "status" / "serve-vptft.json").read_text())
        assert st["policy"] == "vptft" and st["port"] == port and st["version"] == "vptft:7-123"
    finally:
        proc.terminate()
        print(proc.communicate(timeout=10)[0][-2000:])
