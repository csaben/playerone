"""OpenAI VPT as a serve.py backend (recurrent; per-connection transformer memory).

Repo: $VPT_REPO (default ~/Code/player-one/third_party/Video-Pre-Training, github.com/openai/Video-Pre-Training)
Weights: $DATA/vpt/{2x.model, <name>.weights} from openaipublic.blob.core.windows.net/minecraft-rl/models/
Extra deps: `uv sync --extra vpt` (gym3, gym==0.23.1, attrs, opencv-python-headless). minerl is stubbed: VPT
only uses it for an item-name table.

Mapping (VPT recorder conventions, see run_inverse_dynamics_model.py):
  preprocess: full JPEG frame -> RGB -> cv2.resize(128x128, INTER_LINEAR)  (what VPT's agent.py does)
  camera: VPT emits [pitch, yaw] degrees = [dy, dx] * 0.15 (CAMERA_SCALER = 360/2400 at MC default
          sensitivity). We reply counts = degrees / DEG_PER_COUNT, DEG_PER_COUNT from MC_SENSITIVITY.
  keys: forward W, back S, left A, right D, jump SPACE, sneak LSHIFT, sprint LCTRL, inventory E, drop Q,
        hotbar.N N, swapHands F. ESC is suppressed unless cfg {"allow_esc": true}. pickItem dropped.
  mb: [attack, use].
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import types
from pathlib import Path

import av
import numpy as np
import torch

VPT_REPO = Path(os.environ.get("VPT_REPO", "~/Code/player-one/third_party/Video-Pre-Training")).expanduser()
MC_SENSITIVITY = float(os.environ.get("MC_SENSITIVITY", "0.5"))  # options.txt mouseSensitivity
# Minecraft MouseHandler: f = s*0.6+0.2; degrees per raw count = f^3 * 8 * 0.15  (0.15 at s=0.5)
DEG_PER_COUNT = (MC_SENSITIVITY * 0.6 + 0.2) ** 3 * 8 * 0.15
KEYMAP = {"forward": "W", "back": "S", "left": "A", "right": "D", "jump": "SPACE", "sneak": "LSHIFT",
          "sprint": "LCTRL", "inventory": "E", "drop": "Q", "swapHands": "F", "ESC": "ESC",
          **{f"hotbar.{i}": str(i) for i in range(1, 10)}}


ASPECT_MODE = os.environ.get("VPT_ASPECT", "crop")  # crop | pad | stretch  (cfg {"aspect": ...} overrides)


def to_16x9(rgb: np.ndarray, mode: str = "crop") -> np.ndarray:
    """VPT saw 640x360 (16:9) frames. 'crop' = center-crop the long axis (keeps crosshair centered and the
    pixel scale VPT expects); 'pad' = black bars; 'stretch' = leave as is (resize squashes)."""
    h, w = rgb.shape[:2]
    target = 16 / 9
    if mode == "stretch" or abs(w / h - target) < 0.01:
        return rgb
    if mode == "pad":
        if w / h > target:  # too wide -> pad height
            nh = round(w / target)
            top = (nh - h) // 2
            return np.pad(rgb, ((top, nh - h - top), (0, 0), (0, 0)))
        nw = round(h * target)
        left = (nw - w) // 2
        return np.pad(rgb, ((0, 0), (left, nw - w - left), (0, 0)))
    if w / h > target:  # too wide -> crop sides
        nw = round(h * target)
        x0 = (w - nw) // 2
        return rgb[:, x0:x0 + nw]
    nh = round(w / target)  # too tall -> crop top/bottom
    y0 = (h - nh) // 2
    return rgb[y0:y0 + nh]


def _import_vpt():
    if "minerl.herobraine.hero.mc" not in sys.modules:
        m, h, hh = (types.ModuleType(n) for n in ("minerl", "minerl.herobraine", "minerl.herobraine.hero"))
        mc = types.ModuleType("minerl.herobraine.hero.mc")
        mc.MINERL_ITEM_MAP = []
        m.herobraine, h.hero, hh.mc = h, hh, mc
        sys.modules.update({"minerl": m, "minerl.herobraine": h, "minerl.herobraine.hero": hh,
                            "minerl.herobraine.hero.mc": mc})
    if str(VPT_REPO) not in sys.path:
        sys.path.insert(0, str(VPT_REPO))
    import agent as vpt_agent  # noqa: PLC0415
    vpt_agent.validate_env = lambda env: None
    return vpt_agent


class VPTBackend:
    name = "vpt"

    def __init__(self, root: Path, weights: str, device: torch.device, model_file: str = "2x.model"):
        import cv2  # noqa: PLC0415
        self.cv2 = cv2
        A = _import_vpt()
        wdir = root / "vpt"
        wpath = Path(weights) if Path(weights).is_absolute() else wdir / weights
        kw = pickle.load(open(wdir / model_file, "rb"))["model"]["args"]
        self.agent = A.MineRLAgent(None, device=str(device), policy_kwargs=kw["net"]["args"],
                                   pi_head_kwargs=kw["pi_head_opts"])
        self.agent.load_weights(str(wpath))
        self.agent.policy.eval()
        self.device = device
        self.version = f"vpt:{wpath.stem}"
        self.loaded_at = time.time()
        self._first = torch.zeros((1,), dtype=torch.bool, device=device)
        st = self.new_conn()  # warm up
        for _ in range(3):
            self._act(st, np.zeros((128, 128, 3), np.uint8), True)
        print(f"[vpt] loaded {wpath.name} ({sum(p.numel() for p in self.agent.policy.parameters()) / 1e6:.0f}M "
              f"params) deg/count={DEG_PER_COUNT:.4f}", flush=True)

    def follow(self, ckpt_dir: Path, poll_s: float = 2.0) -> None:
        """Serve fine-tuned weights from vpt_train.py ($DATA/vpt-ckpt) instead of the static base."""
        self.ft_dir, self.poll_s, self.name = ckpt_dir, poll_s, "vptft"
        self.version, self.ft_step, self.ft_info = f"{self.version} (base, no fine-tune yet)", 0, None

    async def watch(self):
        import asyncio
        import json
        if not getattr(self, "ft_dir", None):
            return  # static base weights
        seen = None
        while True:
            try:
                info = json.loads((self.ft_dir / "latest.json").read_text())
                if info.get("version") != seen:
                    seen = info["version"]
                    # load straight to GPU off the event loop, then swap = one fast GPU->GPU copy between frames
                    sd = await asyncio.to_thread(torch.load, self.ft_dir / info["path"], map_location=self.device)
                    self.agent.policy.load_state_dict(sd, strict=False)  # VPT weights carry an unused aux_value_head
                    del sd
                    self.version, self.ft_step, self.ft_info = info["version"], int(info["step"]), info
                    self.loaded_at = time.time()
                    print(f"[reload] {self.version} step={self.ft_step} n_frames={info.get('n_frames')}", flush=True)
            except FileNotFoundError:
                pass
            except Exception as e:  # noqa: BLE001
                print(f"[reload] failed: {e!r}", flush=True)
                seen = None
            await asyncio.sleep(self.poll_s)

    def status(self) -> dict:
        ft = getattr(self, "ft_info", None) or {}
        return {"state": "serving", "version": self.version, "step": getattr(self, "ft_step", 0),
                "ckpt_n_frames": ft.get("n_frames"), "fresh": ft.get("fresh"),
                "loaded_at": self.loaded_at, "deg_per_count": DEG_PER_COUNT}

    def new_conn(self) -> dict:
        return {"h": self.agent.policy.initial_state(1)}

    def reset(self, st: dict) -> None:
        st["h"] = self.agent.policy.initial_state(1)

    def _act(self, st: dict, img128: np.ndarray, stochastic: bool) -> dict:
        obs = {"img": torch.from_numpy(img128[None]).to(self.device)}
        with torch.inference_mode():
            ac, st["h"], _ = self.agent.policy.act(obs, self._first, st["h"], stochastic=stochastic)
        ac = {"buttons": ac["buttons"].cpu().numpy(), "camera": ac["camera"].cpu().numpy()}
        return self.agent.action_transformer.policy2env(self.agent.action_mapper.to_factored(ac))

    def step(self, st: dict, frame: av.VideoFrame, cfg: dict, rng: np.random.Generator) -> dict:
        t0 = time.perf_counter()
        rgb = to_16x9(frame.to_ndarray(format="rgb24"), str(cfg.get("aspect", ASPECT_MODE)))
        img = self.cv2.resize(rgb, (128, 128), interpolation=self.cv2.INTER_LINEAR)
        env = self._act(st, img, bool(cfg.get("sample_camera", True)))
        infer_ms = (time.perf_counter() - t0) * 1000
        on = lambda k: bool(np.asarray(env.get(k, 0)).reshape(-1)[0])  # noqa: E731
        keys = [tok for k, tok in KEYMAP.items() if on(k) and (k != "ESC" or cfg.get("allow_esc"))]
        pitch, yaw = np.asarray(env["camera"], np.float64).reshape(-1)[:2]
        return {"keys": keys, "mb": [int(on("attack")), int(on("use"))],
                "cam": [round(yaw / DEG_PER_COUNT, 2), round(pitch / DEG_PER_COUNT, 2)],
                "version": self.version, "infer_ms": round(infer_ms, 2), "step": getattr(self, "ft_step", 0)}
