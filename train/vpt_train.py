"""Continuous VPT fine-tuning on player-one chunks (same loop shape as train.py).

Each sample is a window of up to --seq frames from one chunk, run through VPT with zero initial memory
(VPT's transformer handles (B, T) natively). Loss = -logprob of the human action, masked to human,
non-null frames (VPT's own behavioural_cloning.py skips null actions), weighted by takeover weight.
Labels come from dataset.build_labels (keys, mouse buttons, raw camera counts) converted to VPT env actions:
camera degrees = [dy, dx] * DEG_PER_COUNT, keys via vpt_backend.KEYMAP (inverse).

Writes $DATA/vpt-ckpt/latest.weights (policy state_dict, atomic) + latest.json {version, base, step, ...};
`serve.py --policy vpt --vpt-ft` hot-reloads it. Status: $DATA/status/vpttrain.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

import dataset as D
from dataset import ChunkDataset, data_dir
from vpt_backend import DEG_PER_COUNT, KEYMAP, _import_vpt, to_16x9

FORMAT = "player-one-vpt-ft/v1"
# our KEY_VOCAB token -> VPT env button
TOKEN_TO_VPT = {tok: k for k, tok in KEYMAP.items() if k not in ("ESC", "swapHands")}


class VPTChunkDataset(ChunkDataset):
    """Same chunks/labels/weights as BC, but frames preprocessed like VPT: 16:9 center crop -> cv2 bilinear 128."""

    def __init__(self, root, **kw):
        super().__init__(root, **kw)
        import cv2
        self.cv2 = cv2

    def _cache_path(self, sid, key):
        return self.root / "cache" / sid / f"{key}.vpt128.frames.npy"

    def _decode(self, mp4: Path) -> np.ndarray:
        import av
        out = []
        with av.open(str(mp4)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for fr in c.decode(s):
                rgb = to_16x9(fr.to_ndarray(format="rgb24"), "crop")
                out.append(self.cv2.resize(rgb, (128, 128), interpolation=self.cv2.INTER_LINEAR))
        return np.stack(out) if out else np.zeros((0, 128, 128, 3), np.uint8)


def env_actions(labels: dict) -> dict:
    """Per-frame VPT env action dict (batched arrays) from our labels."""
    n = len(labels["keys"])
    env = {k: np.zeros(n, np.int64) for k in
           ["attack", "back", "forward", "jump", "left", "right", "sneak", "sprint", "use", "drop", "inventory"]
           + [f"hotbar.{i}" for i in range(1, 10)]}
    for j, tok in enumerate(D.KEY_VOCAB):
        if tok in TOKEN_TO_VPT:
            env[TOKEN_TO_VPT[tok]] = np.maximum(env[TOKEN_TO_VPT[tok]], labels["keys"][:, j].astype(np.int64))
    env["attack"] = labels["mb"][:, 0].astype(np.int64)
    env["use"] = labels["mb"][:, 1].astype(np.int64)
    raw = labels["cam_raw"].astype(np.float64)  # (n, 2) = dx, dy counts
    env["camera"] = np.stack([raw[:, 1], raw[:, 0]], 1) * DEG_PER_COUNT  # [pitch, yaw] degrees
    return env


class VPT:
    def __init__(self, root: Path, base: str, device: torch.device, model_file="2x.model"):
        A = _import_vpt()
        kw = pickle.load(open(root / "vpt" / model_file, "rb"))["model"]["args"]
        kw["pi_head_opts"]["temperature"] = float(kw["pi_head_opts"]["temperature"])
        self.agent = A.MineRLAgent(None, device=str(device), policy_kwargs=kw["net"]["args"],
                                   pi_head_kwargs=kw["pi_head_opts"])
        self.agent.load_weights(str(root / "vpt" / base))
        self.policy = self.agent.policy
        self.device = device
        self.mapper, self.tf = self.agent.action_mapper, self.agent.action_transformer
        # NB: ActionTransformer.camera_zero_bin is a *method* in VPT (their null check compares against it)
        self.zero_bin = int(self.tf.discretize_camera(np.zeros((1, 2)))[0, 0])

    def chunk_targets(self, c) -> dict:
        """Cached per chunk: policy action indices + null mask + factored labels for metrics."""
        if not hasattr(c, "_vpt"):
            env = env_actions(c.labels)
            fac = self.tf.env2policy(env)  # {"camera": (n,2) bins, "buttons": (n, len(Buttons.ALL))}
            null = (fac["buttons"] == 0).all(1) & (fac["camera"] == self.zero_bin).all(1)
            pol = self.mapper.from_factored(fac)  # {"buttons": (n,1), "camera": (n,1)}
            c._vpt = {"buttons": pol["buttons"].reshape(-1), "camera": pol["camera"].reshape(-1),
                      "null": null, "fac_buttons": fac["buttons"], "fac_camera": fac["camera"]}
        return c._vpt

    def forward(self, img_u8: torch.Tensor):
        """img (B,T,128,128,3) uint8 -> pi logits dict with (B,T,...) entries."""
        b, t = img_u8.shape[:2]
        first = torch.zeros((b, t), dtype=torch.bool, device=self.device)
        state = self.policy.initial_state(b)
        (pi, _, _), _ = self.policy(obs={"img": img_u8}, first=first, state_in=state)
        return pi

    def forward_long(self, img_u8: torch.Tensor, piece: int = 16):
        """Like forward but in pieces carrying transformer memory (for eval on whole chunks; no grad)."""
        b, t = img_u8.shape[:2]
        state = self.policy.initial_state(b)
        outs = []
        for s in range(0, t, piece):
            x = img_u8[:, s:s + piece]
            first = torch.zeros(x.shape[:2], dtype=torch.bool, device=self.device)
            (pi, _, _), state = self.policy(obs={"img": x}, first=first, state_in=state)
            outs.append(pi)
        return {k: torch.cat([o[k] for o in outs], 1) for k in outs[0]}


def make_batch(vpt: VPT, ds, batch: int, seq: int, rng, recent_frac: float, recent_s: float, dev):
    arrived = np.array([c.arrived for c in ds.chunks])
    recent = np.nonzero(arrived >= arrived.max() - recent_s)[0]
    n_recent = int(round(batch * recent_frac)) if len(recent) < len(ds.chunks) else 0
    cw = np.array([c.w.sum() for c in ds.chunks])
    ci = np.concatenate([
        rng.choice(recent, size=n_recent, p=cw[recent] / cw[recent].sum()) if n_recent else np.zeros(0, int),
        rng.choice(len(ds.chunks), size=batch - n_recent, p=cw / cw.sum())])
    T = min(seq, max(len(ds.chunks[i].labels["keys"]) for i in ci))
    img = np.zeros((batch, T, 128, 128, 3), np.uint8)
    btn = np.zeros((batch, T), np.int64)
    cam = np.zeros((batch, T), np.int64)
    wt = np.zeros((batch, T), np.float32)
    for b, i in enumerate(ci):
        c = ds.chunks[i]
        n = len(c.labels["keys"])
        tg = vpt.chunk_targets(c)
        s0 = int(rng.integers(0, max(n - T, 0) + 1))
        L = min(T, n - s0)
        img[b, :L] = c.frames[s0:s0 + L]
        btn[b, :L], cam[b, :L] = tg["buttons"][s0:s0 + L], tg["camera"][s0:s0 + L]
        w = np.zeros(n, np.float32)
        w[c.valid] = c.w  # human frames (with takeover weight); agent frames stay 0
        w[tg["null"]] = 0  # VPT BC skips null actions
        wt[b, :L] = w[s0:s0 + L]
    t = lambda a: torch.from_numpy(a).to(dev)  # noqa: E731
    return t(img), {"buttons": t(btn[..., None]), "camera": t(cam[..., None])}, t(wt)


@torch.no_grad()
def evaluate(vpt: VPT, chunks, dev, use_amp) -> dict | None:
    """Fresh metrics on chunks the model has not trained on (deterministic action vs human action)."""
    nll_sum = w_sum = 0.0
    fac_b, fac_c, lab_b, lab_c, null_all = [], [], [], [], []
    vpt.policy.eval()
    for c in chunks:
        n = len(c.labels["keys"])
        if n == 0 or not len(c.valid):
            continue
        tg = vpt.chunk_targets(c)
        img = torch.from_numpy(np.ascontiguousarray(c.frames[:n]))[None].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            pi = vpt.forward_long(img)
        pi = {k: v.float() for k, v in pi.items()}
        act = {"buttons": torch.from_numpy(tg["buttons"][:n])[None, :, None].to(dev),
               "camera": torch.from_numpy(tg["camera"][:n])[None, :, None].to(dev)}
        lp = vpt.policy.pi_head.logprob(act, pi).reshape(-1).cpu().numpy()
        idx = c.valid[c.valid < n]
        m = np.zeros(n, bool)
        m[idx] = True
        m &= ~tg["null"][:n]
        nll_sum += float(-lp[m].sum())
        w_sum += float(m.sum())
        smp = vpt.policy.pi_head.sample(pi, deterministic=True)
        fac = vpt.mapper.to_factored({k: v.reshape(-1, 1).cpu().numpy() for k, v in smp.items()})
        fac_b.append(fac["buttons"][idx]); fac_c.append(fac["camera"][idx])
        lab_b.append(tg["fac_buttons"][idx]); lab_c.append(tg["fac_camera"][idx]); null_all.append(tg["null"][idx])
    vpt.policy.train()
    if not fac_b:
        return None
    pb, pc, yb, yc = (np.concatenate(x) for x in (fac_b, fac_c, lab_b, lab_c))
    from lib.actions import Buttons  # VPT repo is on sys.path after _import_vpt()
    names = Buttons.ALL
    col = {k: i for i, k in enumerate(names)}

    def recall(k):
        pos = yb[:, col[k]] > 0
        return float((pb[pos, col[k]] > 0).mean()) if pos.any() else float("nan")

    moving = (yc != vpt.zero_bin).any(1)
    keyish = [k for k in names if k not in ("attack", "use")]
    pos = yb[:, [col[k] for k in keyish]] > 0
    prd = pb[:, [col[k] for k in keyish]] > 0
    r = {"nll": nll_sum / max(w_sum, 1), "keys_recall": float((pos & prd).sum() / max(pos.sum(), 1)),
         "keys_precision": float((pos & prd).sum() / max(prd.sum(), 1)),
         "W_recall": recall("forward"), "attack_recall": recall("attack"), "use_recall": recall("use"),
         "cam_move_acc": float((pc[moving] == yc[moving]).all(1).mean()) if moving.any() else float("nan"),
         "cam_still_acc": float((pc[~moving] == yc[~moving]).all(1).mean()) if (~moving).any() else float("nan"),
         "null_frac": float(np.concatenate(null_all).mean()), "n": int(len(pb))}
    return {k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else (None if isinstance(v, float) else v))
            for k, v in r.items()}


def write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, p)


def save(root: Path, vpt: VPT, args, step: int, ds, fresh) -> dict:
    d = root / "vpt-ckpt"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "latest.weights"
    tmp = d / "latest.weights.tmp"
    torch.save({k: v.detach().cpu() for k, v in vpt.policy.state_dict().items()}, tmp)
    os.replace(tmp, p)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    now = time.time()
    info = {"format": FORMAT, "path": p.name, "base": args.base, "model_file": "2x.model", "step": step,
            "version": f"vptft:{step}-{int(now)}", "sha256": h.hexdigest(), "saved_at": now,
            "n_frames": ds.n_frames, "n_chunks": len(ds.chunks), "n_takeover_frames": ds.n_takeover,
            "deg_per_count": DEG_PER_COUNT, "fresh": fresh,
            "train_args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}}
    write_json(d / "latest.json", info)
    return {"step": step, "saved_at": now, "version": info["version"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--base", default="rl-from-early-game-2x.weights", help="weights file in $DATA/vpt")
    ap.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else "cuda")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=32, help="frames per window (zero initial memory)")
    ap.add_argument("--lr", type=float, default=3e-5, help="VPT's BC script used 1.81e-4 on large data")
    ap.add_argument("--wd", type=float, default=0.039428)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--train-cnn", action="store_true",
                    help="also fine-tune the image encoder (default frozen: its activations are the memory hog)")
    ap.add_argument("--max-epochs", type=float, default=10, help="cap: step*batch*seq/n_frames (0 = none)")
    ap.add_argument("--refresh-s", type=float, default=5)
    ap.add_argument("--save-s", type=float, default=60)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--recent-frac", type=float, default=0.4)
    ap.add_argument("--recent-s", type=float, default=120)
    ap.add_argument("--include-agent", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    root = Path(args.data).expanduser() if args.data else data_dir()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    use_amp = False  # VPT attention asserts Q/K dtypes match; its fp32 state cache breaks bf16 autocast
    rng = np.random.default_rng(0)

    vpt = VPT(root, args.base, dev)
    step = 0
    info_p = root / "vpt-ckpt" / "latest.json"
    if info_p.exists() and not args.no_resume:
        info = json.loads(info_p.read_text())
        if info.get("base") == args.base:
            vpt.policy.load_state_dict(torch.load(root / "vpt-ckpt" / "latest.weights", map_location=dev), strict=False)
            step = int(info["step"])
            print(f"resumed vpt fine-tune at step {step}", flush=True)
    if not args.train_cnn:
        for p in vpt.policy.net.img_process.parameters():
            p.requires_grad_(False)
    params = [p for p in vpt.policy.parameters() if p.requires_grad]
    print(f"trainable params {sum(p.numel() for p in params) / 1e6:.0f}M device={dev}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    vpt.policy.train()

    ds = VPTChunkDataset(root, include_agent=args.include_agent)
    status_p = root / "status" / "vpttrain.json"
    t_refresh, t_save, last_saved, start = 0.0, time.time(), step, step
    first_refresh, fresh, ckpt = True, None, None
    agg, n_agg, t_log = {"loss": 0.0}, 0, time.time()
    while True:
        if time.time() - t_refresh >= args.refresh_s or not ds.chunks:
            added = ds.refresh()
            t_refresh = time.time()
            if added:
                print(f"[refresh] +{added} -> {len(ds.chunks)} chunks {ds.n_frames} frames", flush=True)
                if not first_refresh:
                    fm = evaluate(vpt, ds.chunks[-added:], dev, use_amp)
                    if fm:
                        fresh = {**fm, "at_step": step, "at": time.time()}
                        print("[fresh] " + " ".join(f"{k}={v}" for k, v in fm.items()), flush=True)
                elif step == 0 and len(ds.chunks):  # baseline: the untouched base model on the last chunks
                    fm = evaluate(vpt, ds.chunks[-min(20, len(ds.chunks)):], dev, use_amp)
                    if fm:
                        fresh = {**fm, "at_step": 0, "at": time.time(), "note": "base model, before fine-tuning"}
                        print("[base] " + " ".join(f"{k}={v}" for k, v in fm.items()), flush=True)
            first_refresh = False
        st_common = {"step": step, "n_frames": ds.n_frames, "n_chunks": len(ds.chunks), "fresh": fresh,
                     "ckpt": ckpt, "base": args.base, "updated_at": time.time()}
        if not ds.chunks:
            write_json(status_p, {**st_common, "state": "waiting for data"})
            time.sleep(args.refresh_s)
            continue
        epochs = step * args.batch * args.seq / max(ds.n_frames, 1)
        if args.max_epochs > 0 and not args.steps and epochs >= args.max_epochs:
            if step > last_saved:
                ckpt = save(root, vpt, args, step, ds, fresh)
                last_saved = step
            write_json(status_p, {**st_common, "ckpt": ckpt, "epochs": round(epochs, 1),
                                  "state": f"idle: epoch cap ({epochs:.0f}/{args.max_epochs:g}), waiting for more play"})
            time.sleep(args.refresh_s)
            continue

        img, act, wt = make_batch(vpt, ds, args.batch, args.seq, rng, args.recent_frac, args.recent_s, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            pi = vpt.forward(img)
        pi = {k: v.float() for k, v in pi.items()}
        lp = vpt.policy.pi_head.logprob(act, pi)  # (B,T)
        loss = -(lp * wt).sum() / wt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        if wt.sum() > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
        step += 1
        agg["loss"] += loss.item()
        n_agg += 1
        if step % args.log_every == 0:
            dt = time.time() - t_log
            line = {"loss": round(agg["loss"] / n_agg, 4), "it_s": round(n_agg / dt, 2),
                    "mem_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if dev.type == "cuda" else 0}
            print(f"step {step} " + " ".join(f"{k}={v}" for k, v in line.items()) + f" epochs={epochs:.2f}",
                  flush=True)
            write_json(status_p, {**st_common, "state": "training", "train": line, "epochs": round(epochs, 2)})
            agg, n_agg, t_log = {"loss": 0.0}, 0, time.time()
        done = args.steps and step - start >= args.steps
        if done or time.time() - t_save >= args.save_s:
            ckpt = save(root, vpt, args, step, ds, fresh)
            t_save, last_saved = time.time(), step
            print(f"[ckpt] {ckpt['version']}", flush=True)
        if done:
            break


if __name__ == "__main__":
    main()
