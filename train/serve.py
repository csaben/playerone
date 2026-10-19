"""Policy server: one WebSocket contract, pluggable backends.

  --policy bc   (default, :8962) our small CNN, hot-reloads $DATA/ckpt/latest.pt when ckpt/latest.json changes
  --policy vpt  (:8963)          OpenAI VPT (recurrent, per-connection hidden state); see vpt_backend.py

Client -> server
  binary: one JPEG frame of the game window (any size; each backend preprocesses like its training data)
  text:   {"reset": true}   clear per-connection state (BC frame stack / VPT hidden state)
          {"cfg": {"temperature": 1.0, "key_threshold": 0.5, "sample_camera": true, "sample_keys": false}}
Server -> client, one text reply per frame:
  {"keys": ["W", ...], "mb": [l, r], "cam": [dx, dy], "version": str|null, "infer_ms": float, "step": int|null}
Status: $DATA/status/serve-<name>.json every 2 s (bc also writes serve.json for compatibility).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import deque
from pathlib import Path

import av
import numpy as np
import torch
from websockets.asyncio.server import serve

from dataset import data_dir, frame_to_rgb
from model import load_policy, obs_to_tensor

DEFAULT_PORTS = {"bc": 8962, "vpt": 8963, "vptft": 8964, "nitrogen": 8965}
DEFAULT_CFG = {"temperature": 1.0, "key_threshold": 0.5, "sample_camera": True, "sample_keys": False}
NULL_REPLY = {"keys": [], "mb": [0, 0], "cam": [0, 0], "version": None, "infer_ms": 0.0, "step": None}


class BCBackend:
    """Our CNN. Holds the current model; swapped atomically (single attribute assignment) between frames."""

    name = "bc"

    def __init__(self, ckpt_dir: Path, device: torch.device, poll_s: float = 2.0):
        self.dir = ckpt_dir
        self.device = device
        self.poll_s = poll_s
        self.cur = None          # (model, meta, version, centers) or None
        self.seen_version = None
        self.loaded_at = None

    def _load(self):
        m, meta = load_policy(self.dir / "latest.pt", self.device)
        version = f"{meta['step']}-{int(meta['saved_at'])}"  # from the weights actually loaded
        centers = np.asarray(meta["labels"]["cam_bin_centers"], np.float64)
        a = m.arch_args  # warm up (CUDA kernels/cudnn) off the hot path so the swap is seamless
        with torch.inference_mode():
            for _ in range(3):
                m(torch.zeros(1, 3 * a["stack"], a["img_size"], a["img_size"], device=self.device))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return m, meta, version, centers

    async def watch(self):
        while True:
            try:
                info = json.loads((self.dir / "latest.json").read_text())
                v = info.get("version")
                if v and v != self.seen_version:
                    self.seen_version = v
                    cur = await asyncio.to_thread(self._load)
                    self.cur = cur
                    self.loaded_at = time.time()
                    print(f"[reload] version={cur[2]} step={cur[1]['step']} n_frames={cur[1]['n_frames']}",
                          flush=True)
            except FileNotFoundError:
                pass
            except Exception as e:  # noqa: BLE001
                print(f"[reload] failed: {e!r}", flush=True)
                self.seen_version = None  # retry next poll
            await asyncio.sleep(self.poll_s)

    def status(self) -> dict:
        cur = self.cur
        return {"state": "serving" if cur else "no checkpoint yet", "version": cur[2] if cur else None,
                "step": int(cur[1]["step"]) if cur else None,
                "ckpt_n_frames": cur[1].get("n_frames") if cur else None, "loaded_at": self.loaded_at}

    def new_conn(self) -> dict:
        return {"stack": deque()}

    def reset(self, st: dict) -> None:
        st["stack"].clear()

    def step(self, st: dict, frame: av.VideoFrame, cfg: dict, rng: np.random.Generator) -> dict:
        cur = self.cur
        S = cur[0].arch_args["stack"] if cur else 4
        size = cur[0].arch_args["img_size"] if cur else 128
        img = frame_to_rgb(frame, size)
        stack = st["stack"]
        if not stack:
            stack.extend([img] * S)  # repeat first frame to fill, like training
        else:
            stack.append(img)
        while len(stack) > S:
            stack.popleft()
        if cur is None:
            return dict(NULL_REPLY)
        m, meta, version, centers = cur
        t0 = time.perf_counter()
        obs = np.stack(list(stack))[None]  # (1,S,H,W,3)
        with torch.inference_mode():
            out = m(obs_to_tensor(obs, self.device))
            out = {k: v[0].float().cpu().numpy() for k, v in out.items()}
        infer_ms = (time.perf_counter() - t0) * 1000
        temp = float(cfg.get("temperature", 1.0))
        thr = float(cfg.get("key_threshold", 0.5))

        def binary(logits):
            if cfg.get("sample_keys") and temp > 0:
                return rng.random(logits.shape) < 1 / (1 + np.exp(-logits / temp))
            return 1 / (1 + np.exp(-logits)) >= thr

        def cam_bin(logits):
            if cfg.get("sample_camera", True) and temp > 0:
                z = logits / temp
                p = np.exp(z - z.max())
                return int(rng.choice(len(p), p=p / p.sum()))
            return int(logits.argmax())

        keys = [k for k, on in zip(meta["labels"]["keys"], binary(out["keys"])) if on]
        mb = [int(x) for x in binary(out["mb"])]
        cam = [round(float(centers[cam_bin(out["cam_x"])]), 2), round(float(centers[cam_bin(out["cam_y"])]), 2)]
        return {"keys": keys, "mb": mb, "cam": cam, "version": version, "infer_ms": round(infer_ms, 2),
                "step": int(meta["step"])}


def decode_jpeg(codec: av.CodecContext, data: bytes) -> av.VideoFrame:
    frames = codec.decode(av.Packet(data))
    if not frames:
        raise ValueError("jpeg decode produced no frame")
    return frames[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["bc", "vpt", "nitrogen"], default="bc")
    ap.add_argument("--vpt-ft", action="store_true",
                    help="with --policy vpt: follow fine-tuned weights in $DATA/vpt-ckpt (name vptft, :8964)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=None, help="default 8962 (bc) / 8963 (vpt)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--poll-s", type=float, default=2.0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--vpt-weights", default="rl-from-early-game-2x.weights",
                    help="file in $DATA/vpt (or a path); see vpt_backend.py")
    a = ap.parse_args()
    root = Path(a.data).expanduser() if a.data else data_dir()
    port = a.port or DEFAULT_PORTS["vptft" if a.vpt_ft else a.policy]
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    if a.policy == "nitrogen":
        from nitrogen_backend import NitroGenBackend
        backend = NitroGenBackend(root, dev)
    elif a.policy == "vpt":
        from vpt_backend import VPTBackend
        backend = VPTBackend(root, a.vpt_weights, dev)
        if a.vpt_ft:
            backend.follow(root / "vpt-ckpt", a.poll_s)
    else:
        backend = BCBackend(root / "ckpt", dev, a.poll_s)
    stats = {"n": 0, "ms": 0.0, "t": time.time()}
    clients: dict[int, dict] = {}

    def write_status():
        st = {**backend.status(), "policy": backend.name, "port": port,
              "clients": [{k: v for k, v in c.items() if k != "_t"} for c in clients.values()],
              "updated_at": time.time()}
        names = [f"serve-{backend.name}.json"] + (["serve.json"] if backend.name == "bc" else [])
        for n in names:
            p = root / "status" / n
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(n + ".tmp")
            tmp.write_text(json.dumps(st))
            tmp.replace(p)

    async def status_loop():
        while True:
            try:
                for c in clients.values():  # fps over the last status interval
                    c["fps"] = round((c["frames"] - c["_t"][1]) / max(time.time() - c["_t"][0], 1e-6), 1)
                    c["_t"] = (time.time(), c["frames"])
                write_status()
            except Exception as e:  # noqa: BLE001
                print(f"[status] {e!r}", flush=True)
            await asyncio.sleep(2)

    async def handler(ws):
        codec = av.CodecContext.create("mjpeg", "r")
        cfg = dict(DEFAULT_CFG)
        state = backend.new_conn()
        rng = np.random.default_rng()
        peer = ws.remote_address
        me = {"peer": f"{peer[0]}:{peer[1]}", "connected_at": time.time(), "frames": 0, "errors": 0,
              "last_frame_at": None, "last_reply": None, "fps": 0.0, "_t": (time.time(), 0)}
        clients[id(ws)] = me
        print(f"[conn] {peer} connected", flush=True)
        try:
            async for msg in ws:
                if isinstance(msg, str):
                    try:
                        j = json.loads(msg)
                    except json.JSONDecodeError:
                        print(f"[conn] {peer} bad text message: {msg[:80]!r}", flush=True)
                        continue
                    print(f"[conn] {peer} text {msg[:200]}", flush=True)
                    if j.get("reset"):
                        backend.reset(state)
                    if isinstance(j.get("cfg"), dict):
                        cfg.update(j["cfg"])
                    continue
                try:
                    frame = decode_jpeg(codec, msg)
                except Exception as e:  # noqa: BLE001
                    me["errors"] += 1
                    if me["errors"] <= 3:
                        print(f"[conn] {peer} bad frame ({len(msg)} bytes): {e!r}", flush=True)
                    await ws.send(json.dumps({"error": f"bad frame: {e}"}))
                    continue
                reply = backend.step(state, frame, cfg, rng)
                await ws.send(json.dumps(reply))
                me["frames"] += 1
                me["last_frame_at"] = time.time()
                me["last_reply"] = reply
                if me["frames"] == 1:
                    print(f"[conn] {peer} first frame {len(msg)} bytes -> {reply}", flush=True)
                stats["n"] += 1
                stats["ms"] += reply["infer_ms"]
                if stats["n"] % 100 == 0:
                    dt = time.time() - stats["t"]
                    print(f"[serve] {100 / dt:.1f} fps infer_ms={stats['ms'] / 100:.2f} "
                          f"version={reply['version']}", flush=True)
                    stats["ms"], stats["t"] = 0.0, time.time()
        except Exception as e:  # noqa: BLE001
            print(f"[conn] {peer} error: {e!r}", flush=True)
        finally:
            clients.pop(id(ws), None)
            print(f"[conn] {peer} closed after {me['frames']} frames, {me['errors']} bad "
                  f"(code={ws.close_code} reason={ws.close_reason!r})", flush=True)

    async def run():
        asyncio.create_task(backend.watch())
        asyncio.create_task(status_loop())
        async with serve(handler, a.host, port, max_size=8 * 1024 * 1024, compression=None) as srv:
            print(f"serve[{backend.name}] listening on ws://{a.host}:{port}/ device={dev}", flush=True)
            await srv.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
