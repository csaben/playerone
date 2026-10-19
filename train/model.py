"""Small Impala-style CNN policy: stack of S RGB frames -> keys / mouse buttons / camera x,y bins."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

CKPT_FORMAT = "player-one-bc/v1"


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class ImpalaStage(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.r1, self.r2 = ResBlock(cout), ResBlock(cout)

    def forward(self, x):
        x = F.max_pool2d(self.conv(x), 3, stride=2, padding=1)
        return self.r2(self.r1(x))


class BCPolicy(nn.Module):
    """Input: (B, stack*3, H, W) float in [0,1], frames oldest->newest along channels.
    Output dict: keys (B,K) logits, mb (B,M) logits, cam_x / cam_y (B,bins) logits."""

    def __init__(self, n_keys: int, n_mb: int, cam_bins: int, stack: int, img_size: int,
                 channels=(16, 32, 32), hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.arch_args = dict(n_keys=n_keys, n_mb=n_mb, cam_bins=cam_bins, stack=stack,
                              img_size=img_size, channels=list(channels), hidden=hidden, dropout=dropout)
        self.dropout = dropout
        stages, cin = [], 3 * stack
        for c in channels:
            stages.append(ImpalaStage(cin, c))
            cin = c
        self.stages = nn.Sequential(*stages)
        side = img_size
        for _ in channels:
            side = (side + 1) // 2
        self.fc = nn.Linear(cin * side * side, hidden)
        self.keys = nn.Linear(hidden, n_keys)
        self.mb = nn.Linear(hidden, n_mb)
        self.cam_x = nn.Linear(hidden, cam_bins)
        self.cam_y = nn.Linear(hidden, cam_bins)

    def forward(self, x):
        h = F.relu(self.stages(x)).flatten(1)
        h = F.dropout(F.relu(self.fc(h)), self.dropout, self.training)
        return {"keys": self.keys(h), "mb": self.mb(h), "cam_x": self.cam_x(h), "cam_y": self.cam_y(h)}


def obs_to_tensor(obs_u8, device) -> torch.Tensor:
    """(B, S, H, W, 3) uint8 numpy/tensor -> (B, S*3, H, W) float [0,1]."""
    t = torch.as_tensor(obs_u8, device=device)
    b, s, h, w, c = t.shape
    return t.permute(0, 1, 4, 2, 3).reshape(b, s * c, h, w).float().div_(255.0)


def load_policy(path, device="cpu"):
    """Load a checkpoint written by train.py. Returns (model.eval(), ckpt_dict_without_weights)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    assert ck.get("format") == CKPT_FORMAT, f"unknown checkpoint format {ck.get('format')}"
    m = BCPolicy(**ck["arch_args"]).to(device)
    m.load_state_dict(ck["model"])
    meta = {k: v for k, v in ck.items() if k not in ("model", "optim")}
    return m.eval(), meta
