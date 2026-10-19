"""Continuous behavior-clone trainer.

Default: loop forever - refresh the dataset every --refresh-s, train on random minibatches over
everything seen so far, atomically save $DATA/ckpt/latest.pt every --save-s seconds / --save-steps.
`--steps N` stops after N steps; `--once` = refresh once, train --steps (default 200), save, exit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import ChunkDataset, data_dir, label_config, KEY_VOCAB, MOUSE_BUTTONS, CAM_BINS, IMG_SIZE
from model import BCPolicy, CKPT_FORMAT, obs_to_tensor


LOW_THR = 0.3  # extra recall/precision reported at this sigmoid threshold


def augment(x: torch.Tensor, shift: int, color: float) -> torch.Tensor:
    """Same random translation (+-shift px, border padding) and brightness/contrast jitter for every
    frame of a sample's stack. x: (B, S*3, H, W) in [0,1]."""
    b, _, h, w = x.shape
    if shift > 0:
        t = (torch.randint(-shift, shift + 1, (b, 2), device=x.device).float()
             * torch.tensor([2.0 / w, 2.0 / h], device=x.device))
        theta = torch.zeros(b, 2, 3, device=x.device)
        theta[:, 0, 0] = theta[:, 1, 1] = 1.0
        theta[:, :, 2] = t
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="nearest", padding_mode="border", align_corners=False)
    if color > 0:
        c = 1 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * color       # contrast
        br = (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * color * 0.5  # brightness
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - mean) * c + mean + br).clamp_(0, 1)
    return x


def pos_weights(ds: ChunkDataset, cap: float, dev) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """(1-p)/p per label from all target frames, clamped to [1, cap]. None when disabled."""
    if cap <= 0 or not ds.chunks:
        return None, None
    k = np.concatenate([c.labels["keys"][c.valid] for c in ds.chunks]).mean(0)
    m = np.concatenate([c.labels["mb"][c.valid] for c in ds.chunks]).mean(0)
    f = lambda p: torch.as_tensor(np.clip((1 - p) / np.maximum(p, 1e-6), 1, cap), dtype=torch.float32, device=dev)  # noqa: E731
    return f(k), f(m)


def head_metrics(out, keys, mb, cam) -> dict:
    kp, pos = out["keys"] > 0, keys > 0.5
    kl = torch.sigmoid(out["keys"]) > LOW_THR
    still = cam == CAM_BINS // 2
    px, py = out["cam_x"].argmax(1), out["cam_y"].argmax(1)
    moving = ~(still[:, 0] & still[:, 1])
    return {
        "keys_acc": (kp == pos).float().mean().item(),
        "keys_recall": ((kp & pos).sum() / pos.sum().clamp(min=1)).item(),
        "keys_precision": ((kp & pos).sum() / kp.sum().clamp(min=1)).item(),
        "keys_recall@0.3": ((kl & pos).sum() / pos.sum().clamp(min=1)).item(),
        "keys_precision@0.3": ((kl & pos).sum() / kl.sum().clamp(min=1)).item(),
        "W_recall": ((kp[:, 0] & pos[:, 0]).sum() / pos[:, 0].sum().clamp(min=1)).item(),
        "mb_acc": ((out["mb"] > 0) == (mb > 0.5)).float().mean().item(),
        "cx_acc": (px == cam[:, 0]).float().mean().item(),
        "cy_acc": (py == cam[:, 1]).float().mean().item(),
        # camera accuracy on frames where the mouse actually moved (the still bin dominates otherwise)
        "cam_move_acc": (((px == cam[:, 0]) & (py == cam[:, 1]))[moving].float().mean().item()
                         if moving.any() else float("nan")),
        "cam_still": still.float().mean().item(),
    }


@torch.no_grad()
def evaluate(model, chunks, stack: int, dev, use_amp: bool, bs: int = 256) -> dict | None:
    obs, K, M, C = [], [], [], []
    for c in chunks:
        for i in c.valid:
            obs.append(c.frames[np.clip(np.arange(i - stack + 1, i + 1), 0, None)])
            K.append(c.labels["keys"][i]); M.append(c.labels["mb"][i]); C.append(c.labels["cam"][i])
    if not obs:
        return None
    model.eval()
    outs = []
    for j in range(0, len(obs), bs):
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            o = model(obs_to_tensor(np.stack(obs[j:j + bs]), dev))
        outs.append({k: v.float() for k, v in o.items()})
    model.train()
    out = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
    t = lambda a, dt: torch.as_tensor(np.stack(a), device=dev, dtype=dt)  # noqa: E731
    m = head_metrics(out, t(K, torch.float32), t(M, torch.float32), t(C, torch.int64))
    return {k: round(v, 4) for k, v in m.items()} | {"n": len(obs)}


def write_status(root: Path, **kw) -> None:
    p = root / "status" / "train.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name("train.json.tmp")
    def clean(v):  # NaN/inf -> null so browsers can JSON.parse it
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    tmp.write_text(json.dumps(clean({**kw, "updated_at": time.time()})))
    os.replace(tmp, p)


def save_ckpt(path: Path, model, opt, args, step: int, ds: ChunkDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ck = {
        "format": CKPT_FORMAT,
        "arch": "impala-cnn",
        "arch_args": model.arch_args,
        "labels": label_config(),
        "frame_stack": args.stack,
        "input": {"size": IMG_SIZE, "resize": "plain (no letterbox)", "color": "rgb",
                  "layout": "(B, stack*3, H, W) float32 in [0,1], frames oldest->newest"},
        "outputs": {"keys": "sigmoid logits over labels.keys", "mb": "sigmoid logits over labels.mouse_buttons",
                    "cam_x": "softmax over labels.cam_bins (bin -> counts: labels.cam_bin_centers)",
                    "cam_y": "same as cam_x, +y = mouse down"},
        "step": step,
        "n_frames": ds.n_frames,
        "n_chunks": len(ds.chunks),
        "include_agent": args.include_agent,
        "train_args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))},
        "saved_at": time.time(),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optim": opt.state_dict(),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ck, tmp)
    os.replace(tmp, path)
    # small pointer for the serve side to poll; written only after latest.pt is in place
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    info = {"format": CKPT_FORMAT, "path": path.name, "step": step,
            "version": f"{step}-{int(ck['saved_at'])}", "sha256": h.hexdigest(),
            "mtime": path.stat().st_mtime, "saved_at": ck["saved_at"],
            "n_frames": ds.n_frames, "n_chunks": len(ds.chunks), "n_takeover_frames": ds.n_takeover}
    jt = path.with_name("latest.json.tmp")
    jt.write_text(json.dumps(info))
    os.replace(jt, path.with_name("latest.json"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="data dir (default $DATA or ~/player-one-data)")
    ap.add_argument("--steps", type=int, default=0, help="stop after N steps (0 = forever)")
    ap.add_argument("--once", action="store_true", help="refresh once, train --steps (default 200), save, exit")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--stack", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--refresh-s", type=float, default=5)
    ap.add_argument("--save-s", type=float, default=15)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--include-agent", action="store_true", help="also train on agent-sourced frames")
    ap.add_argument("--recent-frac", type=float, default=0.4, help="share of each batch drawn from recent play")
    ap.add_argument("--recent-s", type=float, default=120, help="recent = chunks arriving within this many s of the newest")
    ap.add_argument("--aug-shift", type=int, default=8, help="random translate +-N px (0 = off)")
    ap.add_argument("--aug-color", type=float, default=0.2, help="brightness/contrast jitter (0 = off)")
    ap.add_argument("--dropout", type=float, default=0.3, help="dropout after the fc layer")
    ap.add_argument("--wd", type=float, default=0.05, help="AdamW weight decay")
    ap.add_argument("--max-epochs", type=float, default=50,
                    help="stop stepping once step*batch/n_frames reaches this; resumes as new play grows the "
                         "data (= at most N passes per frame of play on average). 0 = no cap")
    ap.add_argument("--pos-weight-max", type=float, default=10,
                    help="BCE pos_weight=(1-p)/p per key/button from label freq, clamped to [1,N]. 0 = off")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.once and args.steps == 0:
        args.steps = 200

    root = Path(args.data).expanduser() if args.data else data_dir()
    ckpt_path = root / "ckpt" / "latest.pt"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(args.seed)

    ds = ChunkDataset(root, include_agent=args.include_agent)
    model = BCPolicy(len(KEY_VOCAB), len(MOUSE_BUTTONS), CAM_BINS, args.stack, IMG_SIZE,
                     dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    step = 0
    if ckpt_path.exists() and not args.no_resume:
        try:
            ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
            if ck.get("arch_args") == model.arch_args and ck.get("labels", {}).get("keys") == list(KEY_VOCAB):
                model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["optim"])
                step = int(ck["step"])
                print(f"resumed from {ckpt_path} at step {step}", flush=True)
            else:
                print("checkpoint config differs from current constants; starting fresh", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"could not resume: {e!r}", flush=True)

    print(f"device={dev} data={root} stack={args.stack} batch={args.batch}", flush=True)
    t_refresh = 0.0
    t_save = time.time()
    last_saved_step = step
    start_step = step
    agg: dict[str, float] = {}
    n_agg = 0
    t_log = time.time()
    use_amp = dev.type == "cuda"
    first_refresh = True
    st_kw: dict = {}
    pw_keys = pw_mb = None
    fresh: dict | None = None
    ckpt_info = {"step": step, "saved_at": ckpt_path.stat().st_mtime} if ckpt_path.exists() else None

    while True:
        now = time.time()
        if now - t_refresh >= args.refresh_s or not ds.chunks:
            t0 = time.time()
            added = ds.refresh()
            t_refresh = time.time()
            if added:
                print(f"[refresh] +{added} chunks -> {len(ds.chunks)} chunks, {ds.n_frames} frames, {ds.n_takeover} takeover "
                      f"({t_refresh - t0:.1f}s)", flush=True)
                # "fresh" = how well the current model predicts play it has never trained on (the chunks
                # that just arrived, scored before they enter the pool). Skipped on the startup load
                # after a resume, since those chunks were already trained on.
                if not first_refresh and step > 0:
                    fm = evaluate(model, ds.chunks[-added:], args.stack, dev, use_amp)
                    if fm:
                        fresh = {**fm, "at_step": step, "at": time.time()}
                        print("[fresh] " + " ".join(f"{k}={v:.3f}" for k, v in fm.items()), flush=True)
            if added:
                pw_keys, pw_mb = pos_weights(ds, args.pos_weight_max, dev)
                if pw_keys is not None:
                    print("[pos_weight] " + " ".join(f"{k}={v:.1f}" for k, v in zip(KEY_VOCAB, pw_keys.tolist())
                                                     if v > 1) + f" | mb={[round(v, 1) for v in pw_mb.tolist()]}",
                          flush=True)
            first_refresh = False
        if ds.chunks and args.max_epochs > 0 and not args.steps and \
                step * args.batch / ds.n_frames >= args.max_epochs:
            if step > last_saved_step:
                save_ckpt(ckpt_path, model, opt, args, step, ds)
                t_save, last_saved_step = time.time(), step
                ckpt_info = {"step": step, "saved_at": time.time()}
            ep = step * args.batch / ds.n_frames
            write_status(root, **{**st_kw, "state": f"idle: epoch cap ({ep:.0f}/{args.max_epochs:g}), "
                                                   f"waiting for more play", "step": step,
                                  "epochs": round(ep, 1), "fresh": fresh, "n_frames": ds.n_frames,
                                  "n_chunks": len(ds.chunks)}, ckpt=ckpt_info)
            time.sleep(args.refresh_s)
            continue
        if not ds.chunks:
            if args.once:
                raise SystemExit("no data")
            print("[wait] no complete chunks yet", flush=True)
            write_status(root, state="waiting for data", step=step, n_frames=0, n_chunks=0)
            time.sleep(args.refresh_s)
            continue

        b = ds.sample(args.batch, args.stack, rng, args.recent_frac, args.recent_s)
        x = augment(obs_to_tensor(b["obs"], dev), args.aug_shift, args.aug_color)
        keys = torch.as_tensor(b["keys"], device=dev)
        mb = torch.as_tensor(b["mb"], device=dev)
        cam = torch.as_tensor(b["cam"], device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(x)
        out = {k: v.float() for k, v in out.items()}
        l_keys = F.binary_cross_entropy_with_logits(out["keys"], keys, pos_weight=pw_keys)
        l_mb = F.binary_cross_entropy_with_logits(out["mb"], mb, pos_weight=pw_mb)
        l_cx = F.cross_entropy(out["cam_x"], cam[:, 0])
        l_cy = F.cross_entropy(out["cam_y"], cam[:, 1])
        loss = l_keys + l_mb + 0.5 * (l_cx + l_cy)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        with torch.no_grad():
            m = {"loss": loss.item(), "l_keys": l_keys.item(), "l_mb": l_mb.item(),
                 "l_cx": l_cx.item(), "l_cy": l_cy.item(), **head_metrics(out, keys, mb, cam)}
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        n_agg += 1
        if step % args.log_every == 0:
            dt = time.time() - t_log
            train_m = {k: round(v / n_agg, 4) for k, v in agg.items()}
            ips = n_agg / dt
            s = " ".join(f"{k}={v:.3f}" for k, v in train_m.items())
            print(f"step {step} {s} | {ips:.1f} it/s frames={ds.n_frames}", flush=True)
            agg, n_agg, t_log = {}, 0, time.time()
            st_kw = dict(state="training", step=step, it_s=round(ips, 1), train=train_m, fresh=fresh,
                         n_frames=ds.n_frames, n_chunks=len(ds.chunks), n_takeover_frames=ds.n_takeover,
                         epochs=round(step * args.batch / max(ds.n_frames, 1), 1))
            write_status(root, **st_kw, ckpt=ckpt_info)

        done = args.steps and (step - start_step) >= args.steps
        if done or step - last_saved_step >= args.save_steps or time.time() - t_save >= args.save_s:
            save_ckpt(ckpt_path, model, opt, args, step, ds)
            t_save, last_saved_step = time.time(), step
            ckpt_info = {"step": step, "saved_at": time.time()}
            if st_kw:
                write_status(root, **st_kw, ckpt=ckpt_info)
            print(f"[ckpt] {ckpt_path} step={step} frames={ds.n_frames}", flush=True)
        if done:
            break


if __name__ == "__main__":
    main()
