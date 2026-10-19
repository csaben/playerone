"""Synthetic end-to-end: fake chunks -> HTTP PUT to ingest -> dataset labels -> train a few steps -> load ckpt."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.request
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import dataset as D  # noqa: E402
import ingest  # noqa: E402

FPS = 20
N = 40
W, H = 96, 64  # non-square on purpose


def make_chunk(out: Path, sid: str, idx: int) -> dict[str, Path]:
    key = f"{sid}_c{idx:05d}"
    t0 = idx * 2_000_000
    ts = [t0 + i * 50_000 for i in range(N)]
    mp4 = out / f"{key}.p0.mp4"
    with av.open(str(mp4), "w") as c:
        s = c.add_stream("libx264", rate=FPS)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.time_base = Fraction(1, FPS)
        for i in range(N):
            img = np.zeros((H, W, 3), np.uint8)
            img[..., 0] = (i * 6) % 255
            img[: H // 2, :, 1] = 200 if i % 2 == 0 else 0
            f = av.VideoFrame.from_ndarray(img, format="rgb24")
            f.pts = i
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    p0, ev = [], []
    seq = 0
    for i, t in enumerate(ts):
        keys = ["W"] if i % 2 == 0 else ["A", "LSHIFT", "F"]
        src = "human" if i < N - 5 else "agent:test"
        p0.append({"frame": i, "frame_id": i, "captured_t_us": t, "t_us": t, "heartbeat": False,
                   "keys_held": keys, "axes": {}, "buttons": {},
                   "mouse": {"dx": 999, "dy": 999, "buttons": 1 if 10 <= i < 20 else 0, "wheel": 0},
                   "source": src})
        ev.append({"schema": "agentpc.trajectory/v3", "seq": seq, "t_us": t, "event": "frame", "source_id": 0,
                   "frame_id": i, "captured_t_us": t, "observed_t_us": t, "origin": "captured"})
        seq += 1
        # two physical mouse events in (t_i, t_{i+1}] : total dx=50, dy=-3
        for k, (dx, dy) in enumerate([(20, -1), (30, -2)]):
            ev.append({"schema": "agentpc.trajectory/v3", "seq": seq, "t_us": t + 10_000 + k * 20_000,
                       "event": "raw_input", "source_id": 0, "device": "mouse",
                       "action": {"t_us": t, "keys_held": keys, "axes": {}, "buttons": {},
                                  "mouse": {"dx": dx, "dy": dy, "buttons": 0, "wheel": 0}},
                       "injected": False})
            seq += 1
        # an injected event that must be ignored for human frames
        ev.append({"schema": "agentpc.trajectory/v3", "seq": seq, "t_us": t + 45_000, "event": "raw_input",
                   "source_id": 0, "device": "mouse",
                   "action": {"t_us": t, "keys_held": [], "axes": {}, "buttons": {},
                              "mouse": {"dx": 1000, "dy": 1000, "buttons": 0, "wheel": 0}}, "injected": True})
        seq += 1
        if idx == 0 and i == 5:  # explicit agent->human handoff mid-chunk
            ev.append({"schema": "agentpc.trajectory/v3", "seq": seq, "t_us": t, "event": "handoff",
                       "source_id": 0, "from": {"agent": 7}, "to": "human", "reason": "human_activity"})
            seq += 1
    (out / f"{key}.p0.jsonl").write_text("\n".join(json.dumps(r) for r in p0) + "\n")
    (out / f"{key}.events.jsonl").write_text("\n".join(json.dumps(r) for r in ev) + "\n")
    meta = {"session_id": sid, "chunk": idx, "label": "Minecraft* 1.21", "fps": FPS, "width": W, "height": H,
            "frame_count": N, "sources": ["human"]}
    (out / f"{key}.meta.json").write_text(json.dumps(meta))
    return {suf: out / f"{key}{suf}" for suf in ingest.SUFFIXES}


def put(base: str, sid: str, path: Path) -> dict:
    req = urllib.request.Request(f"{base}/chunks/{sid}/{path.name}", data=path.read_bytes(), method="PUT")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(url: str) -> dict:
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def test_cam_bins_symmetric():
    assert D.cam_encode(0) == D.CAM_BINS // 2
    assert D.cam_encode(1) == D.CAM_BINS // 2  # tiny jitter = no move
    assert D.cam_encode(1e6) == D.CAM_BINS - 1 and D.cam_encode(-1e6) == 0
    b = np.arange(D.CAM_BINS)
    assert (D.cam_encode(D.cam_decode(b)) == b).all()
    assert (D.cam_encode(-D.cam_decode(b)) == b[::-1]).all()


def test_end_to_end(tmp_path):
    data = tmp_path / "data"
    src = tmp_path / "src"
    src.mkdir()
    srv = ingest.serve("127.0.0.1", 0, data)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        sid = "s1760790000-1a2b"
        files = make_chunk(src, sid, 0)
        # upload in client order: meta.json last
        for suf in (".p0.mp4", ".p0.jsonl", ".events.jsonl"):
            assert put(base, sid, files[suf]) == {"ok": True}
        assert get(f"{base}/healthz") == {"ok": True, "sessions": 1, "chunks": 0}
        ds = D.ChunkDataset(data)
        assert ds.refresh() == 0  # incomplete chunk not picked up
        assert put(base, sid, files[".meta.json"]) == {"ok": True}
        assert get(f"{base}/healthz") == {"ok": True, "sessions": 1, "chunks": 1}

        # validation
        bad = urllib.request.Request(f"{base}/chunks/{sid}/evil.sh", data=b"x", method="PUT")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 400
        bad = urllib.request.Request(f"{base}/chunks/..%2f/x.meta.json", data=b"x", method="PUT")
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(bad)

        assert ds.refresh() == 1
        c = ds.chunks[0]
        assert c.frames.shape == (N, 128, 128, 3)
        L = c.labels
        vi = D.KEY_VOCAB.index
        assert L["keys"][0, vi("W")] == 1 and L["keys"][1, vi("A")] == 1 and L["keys"][1, vi("LSHIFT")] == 1
        assert L["keys"][1].sum() == 2  # "F" not in vocab
        assert L["mb"][10, 0] == 1 and L["mb"][9, 0] == 0 and L["mb"][:, 1].sum() == 0
        np.testing.assert_allclose(L["cam_raw"][:N - 5], np.tile([50, -3], (N - 5, 1)))
        assert (L["cam"][0] == D.cam_encode([50, -3])).all()
        assert len(c.valid) == N - 5  # agent frames filtered
        assert D.ChunkDataset(data, include_agent=True).refresh() == 1

        # incremental: second chunk picked up, first not re-decoded
        cache_mtime = next((data / "cache").rglob("*.npy")).stat().st_mtime
        files2 = make_chunk(src, sid, 1)
        for suf in ingest.SUFFIXES:
            put(base, sid, files2[suf])
        assert ds.refresh() == 1 and len(ds.chunks) == 2
        # takeover weighting: chunk0 handoff event at frame 5; chunk0 ends on agent frames, so chunk1
        # frame 0 (human) is an agent->human source flip -> all human frames of chunk1 within 8 s
        w0, w1 = ds.chunks[0].w, ds.chunks[1].w
        assert (w0[:5] == 1).all() and (w0[5:] == D.TAKEOVER_WEIGHT).all()
        assert (w1 == D.TAKEOVER_WEIGHT).all()
        assert ds.n_takeover == (N - 10) + (N - 5)
        assert next((data / "cache").rglob("*c00000*.npy")).stat().st_mtime == cache_mtime
        # desktop (computer-use) recordings are never decoded or trained on by game trainers
        files3 = make_chunk(src, "sdesk", 2)
        m3 = json.loads(files3[".meta.json"].read_text())
        files3[".meta.json"].write_text(json.dumps({**m3, "kind": "desktop"}))
        for suf in ingest.SUFFIXES:
            put(base, "sdesk", files3[suf])
        assert ds.refresh() == 0 and len(ds.chunks) == 2
        assert not (data / "cache" / "sdesk").exists()
        b = ds.sample(8, 4, np.random.default_rng(0))
        assert b["obs"].shape == (8, 4, 128, 128, 3)
    finally:
        srv.shutdown()

    r = subprocess.run([sys.executable, str(HERE / "train.py"), "--data", str(data), "--once", "--steps", "6",
                        "--batch", "8", "--log-every", "2"], capture_output=True, text=True, timeout=600)
    print(r.stdout, r.stderr)
    assert r.returncode == 0, r.stderr
    ck = data / "ckpt" / "latest.pt"
    assert ck.exists()

    import torch
    from model import load_policy, obs_to_tensor
    m, meta = load_policy(ck)
    assert meta["step"] == 6 and meta["frame_stack"] == 4 and meta["labels"]["keys"] == D.KEY_VOCAB
    assert meta["n_frames"] == 2 * (N - 5)
    st = ingest.status(data)
    json.dumps(st, allow_nan=False)
    assert st["ingest"]["chunks"] == 3 and st["ingest"]["frames"] == 3 * N  # ingest counts the desktop chunk too
    assert st["train"]["state"] == "training" and st["train"]["step"] == 6 and "keys_acc" in st["train"]["train"]
    assert st["train"]["ckpt"]["step"] == 6 and st["serve"]["state"] == "down"
    info = json.loads((data / "ckpt" / "latest.json").read_text())
    import hashlib
    assert info["step"] == 6 and info["n_frames"] == 2 * (N - 5) and info["n_takeover_frames"] == 65
    assert info["sha256"] == hashlib.sha256(ck.read_bytes()).hexdigest() and info["version"].startswith("6-")
    with torch.no_grad():
        out = m(obs_to_tensor(b["obs"][:2], "cpu"))
    assert out["keys"].shape == (2, len(D.KEY_VOCAB)) and out["cam_x"].shape == (2, D.CAM_BINS)

    # resume continues the step count
    r = subprocess.run([sys.executable, str(HERE / "train.py"), "--data", str(data), "--once", "--steps", "2",
                        "--batch", "8"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    assert "resumed" in r.stdout
    assert load_policy(ck)[1]["step"] == 8
