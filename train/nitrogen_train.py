"""Continuous NitroGen fine-tuning on player-one controller chunks (same loop shape as vpt_train.py).

Each sample is one frame (NitroGen's frame_per_sample=1, 256x256 cv2 INTER_AREA, as nitrogen_backend) and the
next `action_horizon` (18) gamepad actions on NitroGen's 60 Hz grid, starting `action_shift` (3) steps after the
frame. Our p0.jsonl axes/buttons are level state at ~20 fps, so action k = the latest recorded pad state at
t_i + (shift + k) / 60 s. Loss = NitroGen's own flow-matching MSE (model.forward); takeover frames are drawn more
often, as in BC/VPT. Only human frames whose 0.35 s action window fits inside the chunk are targets; chunks without any
human pad input (keyboard/mouse play) are skipped.

Action vector (tokenizer layout, old_layout=False): [21 buttons in BUTTON_ACTION_TOKENS order, j_left x/y, j_right
x/y], sticks mapped [-1,1] -> [0,1], model y = + down (= -XInput y). Triggers are the LEFT/RIGHT_TRIGGER button
entries. GUIDE and the RIGHT_* alias buttons are masked out of the loss (we never record them).

The SigLIP vision tower is frozen (316M params; NitroGen itself freezes its last layer). The trainable rest (~178M:
DiT, VL self-attention, action encoder/decoder, position embedding) is saved to $DATA/ng-ckpt/latest.weights +
latest.json {version: "ngft:<step>-<ts>", ...}; `serve.py --policy nitrogen --ng-ft` (name ngft, :8966) hot-reloads
it. Status: $DATA/status/ngtrain.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import time
from pathlib import Path

import numpy as np
import torch

import dataset as D
from dataset import ChunkDataset, SkipChunk, data_dir
from nitrogen_backend import BTN_MAP, load_nitrogen

FORMAT = "player-one-ngft/v1"
ENV_HZ = 60.0
UNUSED = {"GUIDE", "RIGHT_BOTTOM", "RIGHT_LEFT", "RIGHT_RIGHT", "RIGHT_UP"}  # never recorded -> masked


class NGChunkDataset(ChunkDataset):
    """Chunks with human gamepad input; frames cached as NitroGen sees them (256x256 INTER_AREA)."""

    def __init__(self, root, label_re: str | None = None, **kw):
        super().__init__(root, **kw)
        import cv2
        self.cv2 = cv2
        self.label_re = re.compile(label_re, re.I) if label_re else None

    def _cache_path(self, sid, key):
        return self.root / "cache" / sid / f"{key}.ng256.frames.npy"

    def _decode(self, mp4: Path) -> np.ndarray:
        import av
        out = []
        with av.open(str(mp4)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for fr in c.decode(s):
                out.append(self.cv2.resize(fr.to_ndarray(format="rgb24"), (256, 256),
                                           interpolation=self.cv2.INTER_AREA))
        return np.stack(out) if out else np.zeros((0, 256, 256, 3), np.uint8)

    def _load_chunk(self, meta_path: Path):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.label_re and not self.label_re.search(str(meta.get("label", ""))):
            raise SkipChunk("label filter")
        d, key = meta_path.parent, meta_path.name[: -len(".meta.json")]
        p0 = D._read_jsonl(d / f"{key}.p0.jsonl")  # cheap pad check before decoding any video
        if not any(pad_active(r) and r.get("source", "human") == "human" for r in p0):
            raise SkipChunk("no gamepad input")
        return super()._load_chunk(meta_path)


def pad_active(r: dict) -> bool:
    ax, bt = r.get("axes") or {}, r.get("buttons") or {}
    return any(abs(float(v or 0)) > 0.15 for v in ax.values()) or any(bool(v) for v in bt.values())


class NG:
    def __init__(self, root: Path, base: str, device: torch.device):
        self.model, self.tok, self.proc, self.cfg = load_nitrogen(root / "nitrogen" / base, device)
        from nitrogen.shared import BUTTON_ACTION_TOKENS  # NitroGen repo is on sys.path after load_nitrogen
        self.device = device
        self.tokens = list(BUTTON_ACTION_TOKENS)
        self.horizon = int(self.tok.action_horizon)
        self.shift = int(self.cfg.modality_cfg.action_shift)
        self.dim = int(self.tok.max_action_dim)
        assert self.dim == len(self.tokens) + 4, (self.dim, len(self.tokens))
        # our pad buttons -> column in the action vector; triggers are button columns too
        col = {n: i for i, n in enumerate(self.tokens)}
        self.btn_cols = [(D.PAD_BUTTONS.index(ours), col[theirs]) for theirs, ours in BTN_MAP.items()]
        self.lt_col, self.rt_col = col["LEFT_TRIGGER"], col["RIGHT_TRIGGER"]
        self.mask = np.ones(self.dim, bool)
        for n in UNUSED:
            self.mask[col[n]] = False

    def chunk_targets(self, c) -> dict:
        """Cached per chunk: (n, horizon, dim) action targets and which frames have a full window."""
        if not hasattr(c, "_ng"):
            ax, bt, t = c.labels["pad_axes"], c.labels["pad_btn"], c.labels["t_us"]
            n = len(t)
            per = np.zeros((n, self.dim), np.float32)  # per recorded frame, in action-vector layout
            for ours, theirs in self.btn_cols:
                per[:, theirs] = bt[:, ours]
            per[:, self.lt_col] = np.clip(ax[:, D.PAD_AXES.index("lt")], 0, 1)
            per[:, self.rt_col] = np.clip(ax[:, D.PAD_AXES.index("rt")], 0, 1)
            sx = lambda k, sign=1: (np.clip(sign * ax[:, D.PAD_AXES.index(k)], -1, 1) + 1) / 2  # noqa: E731
            per[:, -4], per[:, -3], per[:, -2], per[:, -1] = sx("lx"), sx("ly", -1), sx("rx"), sx("ry", -1)
            offs = ((self.shift + np.arange(self.horizon)) / ENV_HZ * 1e6).astype(np.int64)
            q = t[:, None] + offs[None]  # (n, horizon) query times
            idx = np.clip(np.searchsorted(t, q, side="right") - 1, 0, n - 1)  # latest frame at or before
            full = q[:, -1] <= t[-1] if n else np.zeros(0, bool)
            c._ng = {"act": per[idx], "full": full}
        return c._ng

    def images(self, frames_u8: np.ndarray) -> torch.Tensor:
        return self.proc(list(frames_u8), return_tensors="pt")["pixel_values"]

    def batch(self, frames_u8: np.ndarray, actions: np.ndarray | None) -> dict:
        """NitroGen data dict for B single-frame samples (as nitrogen_backend._infer, batched)."""
        pv = self.images(frames_u8)  # (B, C, H, W)
        n = self.cfg.modality_cfg.frame_per_sample
        out = []
        for b in range(len(pv)):
            frames = torch.zeros((n, *pv.shape[1:]), dtype=pv.dtype)
            frames[-1:] = pv[b:b + 1]
            dropped = torch.zeros((n,), dtype=torch.bool)
            dropped[: n - 1] = True
            out.append(self.tok.encode({"frames": frames, "dropped_frames": dropped, "game": None}))
        data = {}
        for k, v in out[0].items():
            if isinstance(v, torch.Tensor):
                data[k] = torch.stack([o[k] for o in out]).to(self.device)
            elif isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(np.stack([o[k] for o in out])).to(self.device)
            else:
                data[k] = [o[k] for o in out]
        if actions is not None:
            B = len(actions)
            data["actions"] = torch.from_numpy(actions).to(self.device)
            data["actions_mask"] = torch.from_numpy(np.broadcast_to(self.mask, actions.shape).copy()).to(self.device)
            data["has_real_action"] = torch.ones(B, dtype=torch.bool, device=self.device)
        return data

    def loss(self, frames_u8, actions) -> torch.Tensor:
        """NitroGen's flow-matching loss (takeover weighting happens in make_batch's sampling)."""
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            return self.model(self.batch(frames_u8, actions))["loss"].float()


def make_batch(ng: NG, ds, batch: int, rng, recent_frac: float, recent_s: float):
    """Frames + action windows. Takeover frames are upweighted via the sampling weights (as BC/VPT)."""
    arrived = np.array([c.arrived for c in ds.chunks])
    recent = np.nonzero(arrived >= arrived.max() - recent_s)[0]
    n_recent = int(round(batch * recent_frac)) if len(recent) < len(ds.chunks) else 0
    cw = np.array([c.w.sum() for c in ds.chunks])
    ci = np.concatenate([
        rng.choice(recent, size=n_recent, p=cw[recent] / cw[recent].sum()) if n_recent else np.zeros(0, int),
        rng.choice(len(ds.chunks), size=batch - n_recent, p=cw / cw.sum())])
    frames = np.empty((batch, 256, 256, 3), np.uint8)
    acts = np.empty((batch, ng.horizon, ng.dim), np.float32)
    for b, i in enumerate(ci):
        c = ds.chunks[i]
        tg = ng.chunk_targets(c)
        i_f = int(c.valid[rng.choice(len(c.valid), p=c.w / c.w.sum())])
        frames[b], acts[b] = c.frames[i_f], tg["act"][i_f]
    return frames, acts


class NGDataset(NGChunkDataset):
    """Targets = human frames with a full action window inside the chunk."""

    def __init__(self, root, ng: NG, **kw):
        super().__init__(root, **kw)
        self.ng = ng

    def _load_chunk(self, meta_path: Path):
        c = super()._load_chunk(meta_path)
        full = self.ng.chunk_targets(c)["full"]
        keep = full[c.valid]
        c.valid, c.w = c.valid[keep], c.w[keep]
        return c


@torch.no_grad()
def evaluate(ng: NG, chunks, rng, max_frames: int = 24) -> dict | None:
    """Fresh metrics on chunks not yet trained on: flow loss (fixed noise) + the sampled first action vs the human's
    (stick MAE in [-1,1] units, button recall/precision, trigger MAE)."""
    pool = [(c, int(i)) for c in chunks for i in c.valid]
    if not pool:
        return None
    pick = [pool[j] for j in rng.choice(len(pool), size=min(max_frames, len(pool)), replace=False)]
    frames = np.stack([c.frames[i] for c, i in pick])
    acts = np.stack([ng.chunk_targets(c)["act"][i] for c, i in pick])
    ng.model.eval()
    g = torch.random.get_rng_state(), torch.cuda.get_rng_state(ng.device) if ng.device.type == "cuda" else None
    torch.manual_seed(0)
    loss = float(ng.loss(frames, acts))
    torch.random.set_rng_state(g[0])
    if g[1] is not None:
        torch.cuda.set_rng_state(g[1], ng.device)
    with torch.inference_mode(), torch.autocast(ng.device.type, dtype=torch.bfloat16,
                                                enabled=ng.device.type == "cuda"):
        out = ng.model.get_action(ng.batch(frames, None), old_layout=False)
    pred = out["action_tensor"].float().cpu().numpy()[:, 0]  # (B, dim), first step of the chunk
    ng.model.train()
    ng.model.set_frozen_modules_to_eval_mode()
    y = acts[:, 0]
    nb = len(ng.tokens)
    bcols = [theirs for _, theirs in ng.btn_cols]
    pb, yb = pred[:, bcols] > 0.5, y[:, bcols] > 0.5
    stick = np.abs((pred[:, nb:] - y[:, nb:]) * 2).mean()  # back to [-1,1] units
    trig = np.abs(pred[:, [ng.lt_col, ng.rt_col]].clip(0, 1) - y[:, [ng.lt_col, ng.rt_col]]).mean()
    moving = np.abs(y[:, nb:nb + 2] * 2 - 1).max(1) > 0.3
    r = {"loss": loss, "stick_mae": float(stick),
         "lstick_mae_moving": float(np.abs((pred[moving, nb:nb + 2] - y[moving, nb:nb + 2]) * 2).mean())
         if moving.any() else float("nan"),
         "btn_recall": float((pb & yb).sum() / max(yb.sum(), 1)) if yb.any() else float("nan"),
         "btn_precision": float((pb & yb).sum() / max(pb.sum(), 1)) if pb.any() else float("nan"),
         "trigger_mae": float(trig), "n": len(pick)}
    return {k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else (None if isinstance(v, float) else v))
            for k, v in r.items()}


def write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, p)


def trainable_state(model) -> dict:
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def save(root: Path, ng: NG, args, step: int, ds, fresh) -> dict:
    d = root / "ng-ckpt"
    d.mkdir(parents=True, exist_ok=True)
    p, tmp = d / "latest.weights", d / "latest.weights.tmp"
    torch.save(trainable_state(ng.model), tmp)
    os.replace(tmp, p)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    now = time.time()
    info = {"format": FORMAT, "path": p.name, "base": args.base, "step": step,
            "version": f"ngft:{step}-{int(now)}", "sha256": h.hexdigest(), "saved_at": now,
            "n_frames": ds.n_frames, "n_chunks": len(ds.chunks), "n_takeover_frames": ds.n_takeover,
            "fresh": fresh, "train_args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}}
    write_json(d / "latest.json", info)
    return {"step": step, "saved_at": now, "version": info["version"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--base", default="ng.pt", help="checkpoint in $DATA/nitrogen")
    ap.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else "cuda")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--train-vision", action="store_true", help="also fine-tune the SigLIP tower (default frozen)")
    ap.add_argument("--label", default=None, help="regex on meta.json label (window title), e.g. 'Arceus'")
    ap.add_argument("--max-epochs", type=float, default=10, help="cap: step*batch/n_frames (0 = none)")
    ap.add_argument("--refresh-s", type=float, default=5)
    ap.add_argument("--save-s", type=float, default=120)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--recent-frac", type=float, default=0.4)
    ap.add_argument("--recent-s", type=float, default=120)
    ap.add_argument("--include-agent", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    # every cached chunk stays memory-mapped (one fd each); a long controller session is >1000 chunks
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    root = Path(args.data).expanduser() if args.data else data_dir()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    rng = np.random.default_rng(0)

    ng = NG(root, args.base, dev)
    step = 0
    info_p = root / "ng-ckpt" / "latest.json"
    if info_p.exists() and not args.no_resume:
        info = json.loads(info_p.read_text())
        if info.get("base") == args.base:
            ng.model.load_state_dict(torch.load(root / "ng-ckpt" / info["path"], map_location=dev), strict=False)
            step = int(info["step"])
            print(f"resumed nitrogen fine-tune at step {step}", flush=True)
    ng.model.set_trainable_parameters(tune_vision_tower=args.train_vision)
    params = [p for p in ng.model.parameters() if p.requires_grad]
    print(f"trainable params {sum(p.numel() for p in params) / 1e6:.0f}M device={dev}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    ng.model.train()
    ng.model.set_frozen_modules_to_eval_mode()

    ds = NGDataset(root, ng, include_agent=args.include_agent, label_re=args.label)
    status_p = root / "status" / "ngtrain.json"
    t_refresh, t_save, last_saved, start = 0.0, time.time(), step, step
    first_refresh, fresh, ckpt = True, None, None
    agg, n_agg, t_log = 0.0, 0, time.time()
    while True:
        if time.time() - t_refresh >= args.refresh_s or not ds.chunks:
            added = ds.refresh()
            t_refresh = time.time()
            if added:
                print(f"[refresh] +{added} -> {len(ds.chunks)} chunks {ds.n_frames} frames", flush=True)
                if not first_refresh:
                    fm = evaluate(ng, ds.chunks[-added:], rng)
                    if fm:
                        fresh = {**fm, "at_step": step, "at": time.time()}
                        print("[fresh] " + " ".join(f"{k}={v}" for k, v in fm.items()), flush=True)
                elif step == 0 and len(ds.chunks):  # baseline: the untouched base model on the last chunks
                    fm = evaluate(ng, ds.chunks[-min(20, len(ds.chunks)):], rng, max_frames=64)
                    if fm:
                        fresh = {**fm, "at_step": 0, "at": time.time(), "note": "base model, before fine-tuning"}
                        print("[base] " + " ".join(f"{k}={v}" for k, v in fm.items()), flush=True)
            first_refresh = False
        st_common = {"step": step, "n_frames": ds.n_frames, "n_chunks": len(ds.chunks), "fresh": fresh,
                     "ckpt": ckpt, "base": args.base, "updated_at": time.time()}
        if not ds.chunks:
            write_json(status_p, {**st_common, "state": "waiting for controller data"})
            time.sleep(args.refresh_s)
            continue
        epochs = step * args.batch / max(ds.n_frames, 1)
        if args.max_epochs > 0 and not args.steps and epochs >= args.max_epochs:
            if step > last_saved:
                ckpt = save(root, ng, args, step, ds, fresh)
                last_saved = step
            write_json(status_p, {**st_common, "ckpt": ckpt, "epochs": round(epochs, 1),
                                  "state": f"idle: epoch cap ({epochs:.0f}/{args.max_epochs:g}), waiting for more play"})
            time.sleep(args.refresh_s)
            continue

        frames, acts = make_batch(ng, ds, args.batch, rng, args.recent_frac, args.recent_s)
        loss = ng.loss(frames, acts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        step += 1
        agg += loss.item()
        n_agg += 1
        if step % args.log_every == 0:
            dt = time.time() - t_log
            line = {"loss": round(agg / n_agg, 4), "it_s": round(n_agg / dt, 2),
                    "mem_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if dev.type == "cuda" else 0}
            print(f"step {step} " + " ".join(f"{k}={v}" for k, v in line.items()) + f" epochs={epochs:.2f}",
                  flush=True)
            write_json(status_p, {**st_common, "state": "training", "train": line, "epochs": round(epochs, 2)})
            agg, n_agg, t_log = 0.0, 0, time.time()
        done = args.steps and step - start >= args.steps
        if done or time.time() - t_save >= args.save_s:
            ckpt = save(root, ng, args, step, ds, fresh)
            t_save, last_saved = time.time(), step
            print(f"[ckpt] {ckpt['version']}", flush=True)
        if done:
            break


if __name__ == "__main__":
    main()
