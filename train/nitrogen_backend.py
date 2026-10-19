"""NVIDIA NitroGen (open gamepad foundation model) as a serve.py backend: replies carry "pad".

Repo: $NITROGEN_REPO (default ~/Code/player-one/third_party/NitroGen, github.com/MineDojo/NitroGen)
Weights: $DATA/nitrogen/ng.pt (huggingface nvidia/NitroGen). Extra deps: `uv sync --extra nitrogen`.

Model: one 256x256 frame (cv2 INTER_AREA, as scripts/play.py) -> a chunk of `action_per_chunk` (18) gamepad
actions meant for a 60 Hz loop. Our clients send ~20 fps and want one reply per frame, so each connection keeps
the current chunk and its start time: reply = chunk[(now - start) * 60]; re-infer on the newest frame when the
chunk is used up. (NitroGen's own play.py freezes the game with a speedhack during inference; we don't.)

Mapping to our pad schema (= agentpc-core Axes/Buttons, XInput conventions):
  sticks: model x as-is; model y is "+ = down" (play.py negates it for vgamepad on Windows) -> ly = -y, ry = -y
  triggers: LEFT_TRIGGER/RIGHT_TRIGGER entries of the button vector, already in [0,1] -> lt/rt
  buttons (> 0.5): SOUTH a, EAST b, WEST x, NORTH y, LEFT_SHOULDER lb, RIGHT_SHOULDER rb, BACK back,
  START start, LEFT_THUMB ls, RIGHT_THUMB rs, DPAD_UP/DOWN/LEFT/RIGHT dpad_u/d/l/r. GUIDE and the
  RIGHT_* aliases are dropped. start/back/guide are suppressed unless cfg {"allow_menu": true} (as play.py).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch

NITROGEN_REPO = Path(os.environ.get("NITROGEN_REPO", "~/Code/player-one/third_party/NitroGen")).expanduser()
ENV_HZ = 60.0
BTN_MAP = {"SOUTH": "a", "EAST": "b", "WEST": "x", "NORTH": "y", "LEFT_SHOULDER": "lb", "RIGHT_SHOULDER": "rb",
           "BACK": "back", "START": "start", "LEFT_THUMB": "ls", "RIGHT_THUMB": "rs", "DPAD_UP": "dpad_u",
           "DPAD_DOWN": "dpad_d", "DPAD_LEFT": "dpad_l", "DPAD_RIGHT": "dpad_r"}
MENU = {"start", "back"}
NULL_PAD = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0, "lt": 0.0, "rt": 0.0, "buttons": []}


def load_nitrogen(ckpt: Path, device: torch.device, dtype=torch.float32):
    """nitrogen.inference_session.load_model, minus the interactive game prompt and prints. fp32 weights +
    bf16 autocast (as upstream); casting the weights to bf16 breaks a masked_scatter inside the model."""
    if str(NITROGEN_REPO) not in sys.path:
        sys.path.insert(0, str(NITROGEN_REPO))
    from transformers import AutoImageProcessor
    from nitrogen.cfg import CkptConfig
    from nitrogen.flow_matching_transformer.nitrogen import NitroGen
    from nitrogen.mm_tokenizers import NitrogenTokenizer
    from nitrogen.shared import PATH_REPO
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = CkptConfig.model_validate(ck["ckpt_config"])
    proc = AutoImageProcessor.from_pretrained(cfg.model_cfg.vision_encoder_name)
    cfg.tokenizer_cfg.training = False
    if cfg.tokenizer_cfg.game_mapping_cfg is not None:
        cfg.tokenizer_cfg.game_mapping_cfg.src_files = [
            x.replace("/mnt/amlfs-02/shared/gaming/gamingvla", str(PATH_REPO))
            for x in cfg.tokenizer_cfg.game_mapping_cfg.src_files]
    tok = NitrogenTokenizer(cfg.tokenizer_cfg)
    model = NitroGen(config=cfg.model_cfg, game_mapping=tok.game_mapping)
    model.load_state_dict(ck["model"])
    del ck
    model.eval()
    tok.eval()
    model.to(dtype=dtype).to(device)
    return model, tok, proc, cfg


class NitroGenBackend:
    name = "nitrogen"

    def __init__(self, root: Path, device: torch.device, ckpt: str = "ng.pt"):
        import cv2
        self.cv2 = cv2
        path = Path(ckpt) if Path(ckpt).is_absolute() else root / "nitrogen" / ckpt
        if device.type == "cuda" and device.index is not None:
            torch.cuda.set_device(device)
        self.model, self.tok, self.proc, self.cfg = load_nitrogen(path, device)
        from nitrogen.shared import BUTTON_ACTION_TOKENS
        self.tokens = list(BUTTON_ACTION_TOKENS)
        self.device = device
        self.version = f"nitrogen:{path.stem}"
        self.loaded_at = time.time()
        self.games = dict(self.tok.game_mapping or {})
        print(f"[nitrogen] loaded {path.name} on {device}; chunk={self.cfg.modality_cfg.action_per_chunk} "
              f"games={len(self.games)}", flush=True)
        st = self.new_conn()
        for _ in range(2):  # warm up
            self._infer(np.zeros((256, 256, 3), np.uint8), None)

    async def watch(self):
        return

    def status(self) -> dict:
        return {"state": "serving", "version": self.version, "step": 0, "ckpt_n_frames": None,
                "loaded_at": self.loaded_at, "action_space": "pad", "n_games": len(self.games)}

    def new_conn(self) -> dict:
        return {"chunk": None, "t0": 0.0}

    def reset(self, st: dict) -> None:
        st["chunk"], st["t0"] = None, 0.0

    def _infer(self, img256: np.ndarray, game: str | None) -> dict:
        pv = self.proc([img256], return_tensors="pt")["pixel_values"]
        n = self.cfg.modality_cfg.frame_per_sample
        frames = torch.zeros((n, *pv.shape[1:]), dtype=pv.dtype)
        frames[-1:] = pv
        dropped = torch.zeros((n,), dtype=torch.bool)
        dropped[: n - 1] = True
        data = self.tok.encode({"frames": frames.to(self.device), "dropped_frames": dropped.to(self.device),
                                "game": game})
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.unsqueeze(0).to(self.device)
            elif isinstance(v, np.ndarray):
                data[k] = torch.tensor(v, device=self.device).unsqueeze(0)
            else:
                data[k] = [v]
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            out = self.tok.decode(self.model.get_action(data, old_layout=False))
        return {k: np.asarray(out[k].squeeze().float().cpu().numpy()) for k in ("j_left", "j_right", "buttons")}

    def step(self, st: dict, frame: av.VideoFrame, cfg: dict, rng: np.random.Generator) -> dict:
        now = time.perf_counter()
        infer_ms = 0.0
        ch = st["chunk"]
        idx = int((now - st["t0"]) * ENV_HZ) if ch is not None else 0
        if ch is None or idx >= len(ch["buttons"]):
            rgb = frame.to_ndarray(format="rgb24")
            img = self.cv2.resize(rgb, (256, 256), interpolation=self.cv2.INTER_AREA)
            t0 = time.perf_counter()
            st["chunk"] = ch = self._infer(img, cfg.get("game"))
            infer_ms = (time.perf_counter() - t0) * 1000
            st["t0"], idx = time.perf_counter(), 0
        jl, jr, bv = ch["j_left"][idx], ch["j_right"][idx], ch["buttons"][idx]
        held = []
        lt = rt = 0.0
        for name, v in zip(self.tokens, bv):
            if name == "LEFT_TRIGGER":
                lt = float(np.clip(v, 0, 1))
            elif name == "RIGHT_TRIGGER":
                rt = float(np.clip(v, 0, 1))
            elif name in BTN_MAP and v > 0.5 and (BTN_MAP[name] not in MENU or cfg.get("allow_menu")):
                held.append(BTN_MAP[name])
        c = lambda x: round(float(np.clip(x, -1, 1)), 3)  # noqa: E731
        pad = {"lx": c(jl[0]), "ly": c(-jl[1]), "rx": c(jr[0]), "ry": c(-jr[1]),
               "lt": round(lt, 3), "rt": round(rt, 3), "buttons": held}
        return {"pad": pad, "version": self.version,  # no keys/mb/cam: absent = sidecar leaves KBM alone
                "infer_ms": round(infer_ms, 2), "step": 0}
