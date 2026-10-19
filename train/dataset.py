"""Incremental dataset over agentpc-format chunks in $DATA/sessions/<sid>/<key>.{p0.mp4,p0.jsonl,events.jsonl,meta.json}.

Decoded frames are cached once per chunk as $DATA/cache/<sid>/<key>.v1.s128.frames.npy (uint8 N,H,W,3,
memory-mapped on load). Labels are recomputed from the jsonl files on every load (cheap), so
editing the vocab/bin constants below takes effect without re-decoding video.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---- label constants (edit freely; checkpoints record them) -------------------------------
KEY_VOCAB: list[str] = ["W", "A", "S", "D", "SPACE", "LSHIFT", "LCTRL", "E", "Q",
                        "1", "2", "3", "4", "5", "6", "7", "8", "9"]
MOUSE_BUTTONS: list[str] = ["left", "right"]  # bit0, bit1 of p0.jsonl mouse.buttons
IMG_SIZE = 128              # frames are plain-resized (no letterbox) to IMG_SIZE x IMG_SIZE RGB
CAM_BINS = 11               # per axis; center bin (5) = no move
CAM_CLIP = 200.0            # raw mouse counts per frame interval, clipped to +-CAM_CLIP
CAM_MU = 10.0               # mu-law strength (VPT-style): fine bins near 0, coarse at the edges
CACHE_VERSION = 1
TAKEOVER_S = 8.0            # human frames within this many seconds after an agent->human handoff ...
TAKEOVER_WEIGHT = 3.0       # ... are sampled this many times more often (DAgger-style corrections)


def cam_encode(x) -> np.ndarray:
    """raw counts -> bin index in [0, CAM_BINS). Symmetric mu-law, center bin = 0 motion."""
    half = (CAM_BINS - 1) / 2
    v = np.clip(np.asarray(x, dtype=np.float64), -CAM_CLIP, CAM_CLIP) / CAM_CLIP
    y = np.sign(v) * np.log1p(CAM_MU * np.abs(v)) / math.log1p(CAM_MU)
    return (np.round(y * half) + half).astype(np.int64)


def cam_decode(b) -> np.ndarray:
    """bin index -> representative raw counts (inverse mu-law at the bin center)."""
    half = (CAM_BINS - 1) / 2
    y = (np.asarray(b, dtype=np.float64) - half) / half
    v = np.sign(y) * (np.power(1.0 + CAM_MU, np.abs(y)) - 1.0) / CAM_MU
    return v * CAM_CLIP


def label_config() -> dict:
    return {"keys": list(KEY_VOCAB), "mouse_buttons": list(MOUSE_BUTTONS), "img_size": IMG_SIZE,
            "cam_bins": CAM_BINS, "cam_clip": CAM_CLIP, "cam_mu": CAM_MU,
            "cam_bin_centers": cam_decode(np.arange(CAM_BINS)).round(2).tolist()}


def data_dir() -> Path:
    return Path(os.environ.get("DATA", "~/player-one-data")).expanduser()


def _read_jsonl(p: Path) -> list[dict]:
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # tolerate a torn last line
    return out


def frame_to_rgb(fr, size: int = IMG_SIZE) -> np.ndarray:
    """THE preprocessing: PyAV VideoFrame -> plain-resized (size,size,3) uint8 RGB (libswscale, PyAV
    default interpolation). Used for training (mp4 frames) and serving (JPEG frames) alike."""
    return fr.to_ndarray(width=size, height=size, format="rgb24")


def decode_frames(mp4: Path, size: int = IMG_SIZE) -> np.ndarray:
    import av
    frames = []
    with av.open(str(mp4)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for fr in c.decode(s):
            frames.append(frame_to_rgb(fr, size))
    if not frames:
        return np.zeros((0, size, size, 3), np.uint8)
    return np.stack(frames)


def build_labels(p0: list[dict], events: list[dict], fps: float) -> dict[str, np.ndarray]:
    """Per-frame labels. Camera for frame i = sum of raw mouse dx/dy with t in (t_i, t_{i+1}]."""
    n = len(p0)
    kidx = {k: j for j, k in enumerate(KEY_VOCAB)}
    keys = np.zeros((n, len(KEY_VOCAB)), np.uint8)
    mb = np.zeros((n, len(MOUSE_BUTTONS)), np.uint8)
    human = np.zeros(n, bool)
    t = np.zeros(n, np.int64)
    for i, r in enumerate(p0):
        t[i] = int(r.get("t_us", 0))
        for k in r.get("keys_held") or []:
            j = kidx.get(str(k).upper())
            if j is not None:
                keys[i, j] = 1
        b = int(((r.get("mouse") or {}).get("buttons")) or 0)
        for j in range(len(MOUSE_BUTTONS)):
            mb[i, j] = (b >> j) & 1
        human[i] = r.get("source", "human") == "human"

    mt, mdx, mdy, minj = [], [], [], []
    for e in events:
        if e.get("event") != "raw_input" or e.get("device") != "mouse":
            continue
        m = ((e.get("action") or {}).get("mouse")) or {}
        dx, dy = m.get("dx", 0) or 0, m.get("dy", 0) or 0
        if dx == 0 and dy == 0:
            continue
        mt.append(int(e.get("t_us", 0)))
        mdx.append(dx)
        mdy.append(dy)
        minj.append(bool(e.get("injected", False)))
    cam_raw = np.zeros((n, 2), np.float32)
    if n and mt:
        mt_a = np.asarray(mt, np.int64)
        order = np.argsort(mt_a, kind="stable")
        mt_a = mt_a[order]
        d = np.stack([np.asarray(mdx, np.float64)[order], np.asarray(mdy, np.float64)[order]], 1)
        inj = np.asarray(minj)[order]
        dt = int(np.median(np.diff(t))) if n > 1 else int(1e6 / max(fps, 1))
        ends = np.append(t[1:], t[-1] + max(dt, 1))
        # window (t_i, end_i]  ->  searchsorted(side="right") on both bounds
        lo = np.searchsorted(mt_a, t, side="right")
        hi = np.searchsorted(mt_a, ends, side="right")
        d_h = np.where(inj[:, None], 0.0, d)  # human frames: physical (non-injected) motion only
        cs_all = np.vstack([np.zeros((1, 2)), np.cumsum(d, 0)])
        cs_h = np.vstack([np.zeros((1, 2)), np.cumsum(d_h, 0)])
        cam_raw = np.where(human[:, None], cs_h[hi] - cs_h[lo], cs_all[hi] - cs_all[lo]).astype(np.float32)
    return {"keys": keys, "mb": mb, "cam": cam_encode(cam_raw), "cam_raw": cam_raw,
            "human": human, "t_us": t}


@dataclass
class Chunk:
    sid: str
    key: str
    frames: np.ndarray            # (N,H,W,3) uint8, mmap
    labels: dict
    valid: np.ndarray             # frame indices usable as targets
    w: np.ndarray                 # per-valid-frame sampling weight (takeover frames upweighted)
    arrived: float = 0.0          # meta.json mtime (when the chunk finished uploading)
    added: float = field(default_factory=time.time)


class SkipChunk(Exception):
    """Chunk deliberately excluded (not an error)."""


class ChunkDataset:
    # meta.json "kind" (default "game"). Desktop/computer-use recordings ("desktop") are private and are
    # never used by the game trainers unless a caller passes kinds explicitly.
    kinds: tuple[str, ...] = ("game",)

    def __init__(self, root: Path | None = None, include_agent: bool = False, size: int = IMG_SIZE):
        self.root = Path(root) if root else data_dir()
        self.include_agent = include_agent
        self.size = size
        self.chunks: list[Chunk] = []
        self._seen: set[str] = set()
        self._fails: dict[str, int] = {}
        self._handoffs: dict[str, list[int]] = {}   # sid -> agent->human takeover t_us
        self._last_human: dict[str, bool] = {}      # sid -> last frame of previous chunk was human?

    @property
    def n_frames(self) -> int:
        return int(sum(len(c.valid) for c in self.chunks))

    @property
    def n_takeover(self) -> int:
        return int(sum(int((c.w > 1).sum()) for c in self.chunks))

    def _decode(self, mp4: Path) -> np.ndarray:
        """Override in subclasses that need different preprocessing (e.g. VPT)."""
        return decode_frames(mp4, self.size)

    def _cache_path(self, sid: str, key: str) -> Path:
        return self.root / "cache" / sid / f"{key}.v{CACHE_VERSION}.s{self.size}.frames.npy"

    def _load_chunk(self, meta_path: Path) -> Chunk:
        sid = meta_path.parent.name
        key = meta_path.name[: -len(".meta.json")]
        d = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("kind", "game") not in self.kinds:
            raise SkipChunk(f"kind={meta.get('kind')}")
        cp = self._cache_path(sid, key)
        if not cp.exists():
            arr = self._decode(d / f"{key}.p0.mp4")
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_name(cp.name[:-4] + ".tmp.npy")
            np.save(tmp, arr)
            os.replace(tmp, cp)
        frames = np.load(cp, mmap_mode="r")
        p0 = _read_jsonl(d / f"{key}.p0.jsonl")
        ev_path = d / f"{key}.events.jsonl"
        events = _read_jsonl(ev_path) if ev_path.exists() else []
        n = min(len(frames), len(p0))
        labels = build_labels(p0[:n], events, float(meta.get("fps") or 20))
        mask = np.ones(n, bool) if self.include_agent else labels["human"]
        valid = np.nonzero(mask)[0]
        # takeovers = explicit handoff events (from {"agent":id} to "human") + agent->human source flips.
        # Chunks load in arrival order, so handoffs from earlier chunks of the session are known.
        ho = self._handoffs.setdefault(sid, [])
        for e in events:
            if e.get("event") == "handoff" and e.get("to") == "human" and isinstance(e.get("from"), dict):
                ho.append(int(e.get("t_us", 0)))
        hum = labels["human"]
        if n:
            prev = np.concatenate([[self._last_human.get(sid, True)], hum[:-1]])
            ho.extend(int(t) for t in labels["t_us"][hum & ~prev])
            self._last_human[sid] = bool(hum[-1])
        w = np.ones(len(valid), np.float64)
        if ho and len(valid):
            hs = np.sort(np.asarray(ho, np.int64))
            tv = labels["t_us"][valid]
            j = np.searchsorted(hs, tv, side="right") - 1  # latest takeover at or before each frame
            since = np.where(j >= 0, tv - hs[np.clip(j, 0, None)], np.iinfo(np.int64).max)
            w[(since <= TAKEOVER_S * 1e6) & hum[valid]] = TAKEOVER_WEIGHT
        labels["takeover"] = np.zeros(n, bool)
        labels["takeover"][valid[w > 1]] = True
        return Chunk(sid, key, frames, labels, valid, w, meta_path.stat().st_mtime)

    def refresh(self) -> int:
        """Pick up newly completed chunks (meta.json present). Returns number added."""
        sess = self.root / "sessions"
        if not sess.is_dir():
            return 0
        metas = sorted(sess.glob("*/*.meta.json"), key=lambda p: p.stat().st_mtime)
        added = 0
        for mp in metas:
            ident = f"{mp.parent.name}/{mp.name}"
            if ident in self._seen or self._fails.get(ident, 0) >= 3:
                continue
            try:
                c = self._load_chunk(mp)
            except SkipChunk:
                self._seen.add(ident)  # e.g. desktop/computer-use recordings: never decoded, never trained on
                continue
            except Exception as e:  # noqa: BLE001
                self._fails[ident] = self._fails.get(ident, 0) + 1
                print(f"[dataset] failed {ident}: {e!r}", file=sys.stderr, flush=True)
                continue
            self._seen.add(ident)
            if len(c.valid):
                self.chunks.append(c)
                added += 1
        return added

    def sample(self, batch: int, stack: int, rng: np.random.Generator,
               recent_frac: float = 0.4, recent_s: float = 120.0) -> dict[str, np.ndarray]:
        """Random minibatch. `recent_frac` of the batch comes from chunks that arrived within `recent_s`
        of the newest chunk (the last ~2 min of play, even after play stops); the rest from everything.
        Within a pool, frames are drawn proportional to their weight (takeover frames upweighted)."""
        arrived = np.array([c.arrived for c in self.chunks])
        recent = np.nonzero(arrived >= arrived.max() - recent_s)[0]
        n_recent = int(round(batch * recent_frac)) if len(recent) < len(self.chunks) else 0
        cw = np.array([c.w.sum() for c in self.chunks])
        ci = np.concatenate([
            rng.choice(recent, size=n_recent, p=cw[recent] / cw[recent].sum()) if n_recent else np.zeros(0, int),
            rng.choice(len(self.chunks), size=batch - n_recent, p=cw / cw.sum()),
        ])
        obs = np.empty((batch, stack, self.size, self.size, 3), np.uint8)
        keys = np.empty((batch, len(KEY_VOCAB)), np.float32)
        mb = np.empty((batch, len(MOUSE_BUTTONS)), np.float32)
        cam = np.empty((batch, 2), np.int64)
        for b, c_i in enumerate(ci):
            c = self.chunks[c_i]
            i = int(c.valid[rng.choice(len(c.valid), p=c.w / c.w.sum())])
            idx = np.clip(np.arange(i - stack + 1, i + 1), 0, None)  # repeat first frame at chunk start
            obs[b] = c.frames[idx]
            keys[b] = c.labels["keys"][i]
            mb[b] = c.labels["mb"][i]
            cam[b] = c.labels["cam"][i]
        return {"obs": obs, "keys": keys, "mb": mb, "cam": cam}
