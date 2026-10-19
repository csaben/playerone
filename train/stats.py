"""Print per-head label stats for the chunks in $DATA (or --data). Usage: uv run python stats.py [--last N]"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import dataset as D


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--last", type=int, default=0, help="only the newest N chunks")
    ap.add_argument("--include-agent", action="store_true")
    a = ap.parse_args()
    ds = D.ChunkDataset(Path(a.data).expanduser() if a.data else None, include_agent=a.include_agent)
    ds.refresh()
    chunks = ds.chunks[-a.last:] if a.last else ds.chunks
    print(f"{len(chunks)} chunks, {sum(len(c.valid) for c in chunks)} target frames")
    if not chunks:
        return
    for c in chunks[:3]:
        L = c.labels
        dt = np.diff(L["t_us"]) / 1000 if len(L["t_us"]) > 1 else np.array([0])
        print(f"  {c.sid}/{c.key}: frames={len(c.frames)} labels={len(L['keys'])} human={int(L['human'].sum())} "
              f"dt_ms med={np.median(dt):.1f} max={dt.max():.1f}")
    keys = np.concatenate([c.labels["keys"][c.valid] for c in chunks])
    mb = np.concatenate([c.labels["mb"][c.valid] for c in chunks])
    cam = np.concatenate([c.labels["cam"][c.valid] for c in chunks])
    raw = np.concatenate([c.labels["cam_raw"][c.valid] for c in chunks])
    print("key freq:", " ".join(f"{k}={f:.3f}" for k, f in zip(D.KEY_VOCAB, keys.mean(0)) if f > 0) or "(none)")
    print("mouse:", " ".join(f"{k}={f:.3f}" for k, f in zip(D.MOUSE_BUTTONS, mb.mean(0))))
    for ax, name in ((0, "cam_x"), (1, "cam_y")):
        h = np.bincount(cam[:, ax], minlength=D.CAM_BINS)
        r = np.abs(raw[:, ax])
        print(f"{name} bins: {h.tolist()}  still={h[D.CAM_BINS // 2] / h.sum():.3f}  "
              f"|raw| p50={np.percentile(r, 50):.1f} p90={np.percentile(r, 90):.1f} p99={np.percentile(r, 99):.1f} "
              f"max={r.max():.0f} clipped={(r > D.CAM_CLIP).mean():.3f}")


if __name__ == "__main__":
    main()
