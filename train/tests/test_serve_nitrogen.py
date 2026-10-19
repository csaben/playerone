"""NitroGen backend over the WS contract (replies carry "pad"). Skipped unless repo + weights are present."""
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

import dataset as D  # noqa: E402

DATA = Path(os.environ.get("DATA", "~/player-one-data")).expanduser()
REPO = Path(os.environ.get("NITROGEN_REPO", "~/Code/player-one/third_party/NitroGen")).expanduser()
pytestmark = pytest.mark.skipif(not ((DATA / "nitrogen" / "ng.pt").exists() and REPO.exists()),
                                reason="NitroGen repo/weights not present")


def test_serve_nitrogen(tmp_path):
    (tmp_path / "nitrogen").mkdir()
    (tmp_path / "nitrogen" / "ng.pt").symlink_to(DATA / "nitrogen" / "ng.pt")
    port = free_port()
    # CPU so the test never competes with the live servers for GPU memory
    proc = subprocess.Popen([sys.executable, str(HERE / "serve.py"), "--policy", "nitrogen", "--cpu", "--host",
                             "127.0.0.1", "--port", str(port), "--data", str(tmp_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(900):
            try:
                ws = connect(f"ws://127.0.0.1:{port}/", open_timeout=5)
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise AssertionError("server did not start")
        with ws:
            ws.send(json.dumps({"reset": True}))
            for i in range(3):
                ws.send(jpeg(320, 180, seed=i))
                r = json.loads(ws.recv(timeout=300))
                assert set(r) == {"pad", "version", "infer_ms", "step"}, r  # no keys/mb/cam: leave KBM alone
                p = r["pad"]
                assert set(p) == set(D.PAD_AXES) | {"buttons"}
                assert all(-1 <= p[k] <= 1 for k in ("lx", "ly", "rx", "ry")) and 0 <= p["lt"] <= 1 and 0 <= p["rt"] <= 1
                assert set(p["buttons"]) <= set(D.PAD_BUTTONS) and not ({"start", "back"} & set(p["buttons"]))
        time.sleep(2.5)
        st = json.loads((tmp_path / "status" / "serve-nitrogen.json").read_text())
        assert st["policy"] == "nitrogen" and st["action_space"] == "pad" and st["port"] == port
    finally:
        proc.terminate()
        print(proc.communicate(timeout=20)[0][-2000:])
