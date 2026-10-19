# player-one/train: cradle ingest and continuous BC trainer

Receives agentpc-format recording chunks from the player-one dock and continuously trains a small behavior-clone policy for Minecraft.

## Running (cradle)

```bash
# deploy from inari (player-one has no git remote)
cd /d/Code/player-one/train && tar cf - pyproject.toml *.py tests README.md | ssh arelius@100.80.98.4 'mkdir -p ~/Code/player-one/train && cd ~/Code/player-one/train && tar xf -'

# on cradle
cd ~/Code/player-one/train && uv sync
uv run pytest -q -s                      # synthetic end-to-end test
tmux new-session -d -s p1train -n ingest "uv run python -u ingest.py --port 8961 2>&1 | tee -a ~/player-one-data/ingest.log"
tmux new-window  -t p1train    -n train  "uv run python -u train.py 2>&1 | tee -a ~/player-one-data/train.log"
```

- `$DATA` defaults to `~/player-one-data` and holds `sessions/`, `cache/`, `ckpt/latest.pt`, and the logs.
- `train.py` runs forever by default. `--steps N` stops after N steps. `--once` refreshes the dataset once, trains `--steps` steps (default 200), saves and exits. Other flags: `--batch 64 --stack 4 --lr 3e-4 --refresh-s 5 --save-s 15 --save-steps 1000 --include-agent --recent-frac 0.4 --recent-s 120 --no-resume`.
- **Recent share.** `--recent-frac` (0.4) of each batch is drawn from chunks that arrived within `--recent-s` (120 s) of the newest chunk. The rest is drawn from everything.
- **Takeovers.** Human frames within `TAKEOVER_S` = 8 s after an agent-to-human takeover get `TAKEOVER_WEIGHT` = 3x sampling weight. A takeover is either a `handoff` event with `from: {"agent": id}, to: "human"` or a frame whose `source` flips from agent to human.
- **Checkpoint pointer.** After each atomic write of `latest.pt`, the trainer atomically writes `ckpt/latest.json`: `{format, path, step, version: "<step>-<unix_s>", sha256, mtime, saved_at, n_frames, n_chunks, n_takeover_frames}`. The serve side should poll this file and only reload `latest.pt` when `version` changes.
- The trainer resumes from `ckpt/latest.pt` if the architecture and key vocab match. Otherwise it starts fresh and overwrites it.

## Ingest contract (port **8961**; 8960 is taken on cradle by ml-dash)

- `PUT /chunks/<session_id>/<filename>` with the raw body writes `$DATA/sessions/<session_id>/<filename>` atomically (tmp + rename). `session_id` and `filename` must match `^[A-Za-z0-9._-]+$`. `filename` must end with `.p0.mp4`, `.p0.jsonl`, `.events.jsonl` or `.meta.json`. Returns 200 `{"ok":true}`. On error it returns 400/404/411/413 with `{"ok":false,"error":...}`.
- `GET /healthz` returns `{"ok":true,"sessions":N,"chunks":M}`, where chunks = number of `*.meta.json` files.
- The client uploads `.meta.json` **last**. A chunk counts as complete only once its meta exists.

## Labels (constants in `dataset.py`)

| head | definition |
|---|---|
| obs | the last `stack`=4 frames (oldest→newest; repeats frame 0 at a chunk start), plain-resized to 128x128 RGB without letterbox |
| keys (BCE) | multi-hot of `p0.jsonl keys_held` over `KEY_VOCAB = W A S D SPACE LSHIFT LCTRL E Q 1 2 3 4 5 6 7 8 9`. Other keys are ignored. |
| mouse buttons (BCE) | `left` = bit0 and `right` = bit1 of `p0.jsonl mouse.buttons` (level state) |
| camera x / y (11-way CE each) | sum of `events.jsonl` `raw_input` mouse `action.mouse.dx/dy` with event `t_us` in (t_i, t_{i+1}], where t is the p0.jsonl `t_us`. The last frame of a chunk uses t_i + median frame dt. Human frames only sum `injected:false` events. The per-frame `p0.jsonl mouse.dx/dy` is not used. |

Camera quantization uses a VPT-style mu-law: clip to ±`CAM_CLIP`=200 counts per frame interval, normalize to [-1,1], then y = sign·ln(1+μ|v|)/ln(1+μ) with `CAM_MU`=10 and bin = round(5·y)+5. That gives `CAM_BINS`=11 bins, with bin 5 meaning no move. Bin-center decode (counts), bins 0..10: `-200, -116.2, -64.3, -32.2, -12.3, 0, 12.3, 32.2, 64.3, 116.2, 200`. Motion under about 5.4 counts lands in the no-move bin. The same values are stored in the checkpoint at `labels.cam_bin_centers`. +x means right and +y means down (raw mouse convention).

By default only frames with `source == "human"` become training targets (`--include-agent` adds agent frames).

## Dataset

`ChunkDataset.refresh()` scans `$DATA/sessions/*/*.meta.json` and loads only the chunks it hasn't seen. Each mp4 is decoded once with PyAV into `$DATA/cache/<sid>/<key>.v1.s128.frames.npy` (uint8, memory-mapped). Labels are rebuilt from the jsonl files on every load, so editing the vocab or bin constants only needs a trainer restart, not a re-decode. A chunk that fails to load is retried on up to 3 refreshes. Sampling uses the recent share and takeover weights described above.

Disk use is about 49 KB per cached frame, roughly 3.5 GB per recorded hour at 20 fps. Cradle's root disk had only about 41 GB free on 2026-10-18.

## Model and checkpoint

`model.py` defines an Impala CNN: 3 stages (16, 32, 32 channels), each a conv, a 3x3/2 maxpool and 2 residual blocks. Then fc 256 feeds four heads: `keys`, `mb`, `cam_x` and `cam_y`. Loss = BCE(keys) + BCE(mb) + 0.5·(CE(cam_x) + CE(cam_y)), trained with AdamW 3e-4 and bf16 autocast on CUDA.

`$DATA/ckpt/latest.pt` is written atomically (tmp + `os.replace`) as a `torch.save` dict:

```python
{
  "format": "player-one-bc/v1", "arch": "impala-cnn",
  "arch_args": {"n_keys", "n_mb", "cam_bins", "stack", "img_size", "channels", "hidden"},
  "labels": {"keys", "mouse_buttons", "img_size", "cam_bins", "cam_clip", "cam_mu", "cam_bin_centers"},
  "frame_stack": 4,
  "input": {"size": 128, "resize": "plain (no letterbox)", "color": "rgb",
            "layout": "(B, stack*3, H, W) float32 in [0,1], frames oldest->newest"},
  "outputs": {...},  # human-readable head descriptions
  "step", "n_frames", "n_chunks", "include_agent", "saved_at",
  "model": state_dict, "optim": optimizer state_dict,
}
```

Load it with `from model import load_policy, obs_to_tensor; m, meta = load_policy(path, device)`. `obs_to_tensor` takes a `(B, S, H, W, 3)` uint8 array.

## Logs

Every 50 steps the trainer prints `loss`, the per-head losses, `keys_acc` (elementwise), `keys_recall` (recall on held keys), `mb_acc`, `cx_acc`/`cy_acc`, and `cam_still` (fraction of no-move targets, the baseline for camera accuracy). It also prints it/s and the frame count.

## Policy server (`serve.py`, tmux window `serve`, port 8962)

The server is a WebSocket at `ws://100.80.98.4:8962/`. It loads a checkpoint only after `ckpt/latest.json` changes; it polls that file every 2 s. Loading and warm-up happen in a background thread, and the new model is swapped in between frames.

- **Client → server, binary.** One JPEG frame per message, at any size; the client sends ≤320 px wide. It is decoded with PyAV mjpeg and passed through the same `dataset.frame_to_rgb` (libswscale plain resize to 128x128 RGB) used for training. Each connection keeps a stack of the last `stack` frames, and the first frame is repeated to fill it, as in training.
- **Client → server, text.** `{"reset":true}` clears the stack. `{"cfg":{"temperature":1.0,"key_threshold":0.5,"sample_camera":true,"sample_keys":false}}` updates sampling for that connection.
- **Server → client.** One text reply per frame: `{"keys":[...],"mb":[l,r],"cam":[dx,dy],"version":"<step>-<unix_s>","infer_ms":..,"step":..}`. `cam` holds bin-center counts. The camera bin is sampled from softmax(logits/T) when `sample_camera` is set and T>0, and is argmax otherwise. Keys and mouse buttons are sigmoid ≥ `key_threshold`, or Bernoulli samples at temperature T when `sample_keys` is set. Before any checkpoint exists, the reply is `{"keys":[],"mb":[0,0],"cam":[0,0],"version":null,...}`.
- **Logging.** The server prints one line per 100 frames: fps, mean infer ms and version.

## VPT backend (`serve.py --policy vpt`, tmux window `vpt`, port 8963)

This uses OpenAI VPT behind the same WebSocket contract. Install with `uv sync --extra vpt`. The repo lives at `~/Code/player-one/third_party/Video-Pre-Training` (override with `$VPT_REPO`). Weights go in `$DATA/vpt` (`2x.model`, `foundation-model-2x`, `bc-early-game-2x`, `rl-from-early-game-2x`); select one with `--vpt-weights`.

- **minerl:** VPT imports it, but only uses it for an item-name table, so it is stubbed.
- **Preprocessing:** the full JPEG is converted to RGB and centre-cropped to 16:9 (`$VPT_ASPECT` or `cfg.aspect`: `crop`, `pad` or `stretch`; default `crop`), then `cv2.resize(128x128, INTER_LINEAR)`, as VPT's `agent.py` does.
- **State:** the hidden state (128-step transformer memory) is kept per connection, and `{"reset":true}` resets it.
- **Camera:** VPT outputs `[pitch, yaw]` in degrees, which is `[dy, dx]·0.15` at MC's default sensitivity. The reply is `cam = [yaw, pitch] / DEG_PER_COUNT`, where `DEG_PER_COUNT = (0.6·s+0.2)³·8·0.15` and `s = $MC_SENSITIVITY` (default 0.5, giving 0.15).
- **Keys:** forward→W, back→S, left→A, right→D, jump→SPACE, sneak→LSHIFT, sprint→LCTRL, inventory→E, drop→Q, hotbar.N→N, swapHands→F. ESC is suppressed unless `cfg.allow_esc` is set. `mb = [attack, use]`.
- **Sampling:** `cfg.sample_camera=false` makes the policy deterministic (argmax).
- **Speed:** about 6–7 ms per step on the 3090 and 1.9 GB of GPU memory.

Each server writes `$DATA/status/serve-<policy>.json`, and `/status` returns all of them under `models`.

## VPT fine-tuning (`vpt_train.py`, tmux window `vpttrain`, GPU 1) and `vptft` (port 8964)

The fine-tuning loop has the same shape as `train.py`: a 5 s refresh, recent share, takeover weights and an epoch cap (`--max-epochs 10`, counted as `step·batch·seq / n_frames`).

- **Frames:** 16:9 centre crop, then cv2 bilinear resize to 128, cached as `cache/<sid>/<key>.vpt128.frames.npy`.
- **Labels:** built from the BC labels:
  - keys via the inverse of `vpt_backend.KEYMAP`
  - `attack`/`use` from the mouse buttons
  - camera degrees = `[dy, dx]·DEG_PER_COUNT`
  - then VPT's `env2policy` and `from_factored`
- **Training:** windows of `--seq 32` frames start from zero memory, `--batch 4`, AdamW with lr 3e-5 and wd 0.039, gradient clip 5, fp32. bf16 autocast breaks VPT's attention dtype assert.
- **Loss mask:** the loss is −logprob, masked to human frames that are not null (as in VPT's BC). Note that VPT's `ActionTransformer.camera_zero_bin` is a method, so the zero bin is computed explicitly.
- **Image encoder:** frozen by default, because its activations are the memory hog. That leaves 225M trainable params, peaking at about 5.3 GB with 8 steps/s on a 3090. `--train-cnn` unfreezes it.
- **Metrics:** on startup it evaluates the base model on the last 20 chunks, labelled `[base]`. After that, "fresh" metrics score new chunks before they are trained on: nll, keys recall/precision, W/attack/use recall, `cam_move_acc` and `cam_still_acc`.
- **Checkpoint:** `$DATA/vpt-ckpt/latest.weights` (policy state_dict) and `latest.json` (`{version: "vptft:<step>-<ts>", base, step, n_frames, fresh, ...}`).
- **Serving:** `serve.py --policy vpt --vpt-ft` (tmux `vptft`, :8964) hot-reloads it. The weights load straight to the GPU in a thread, then the server swaps them in between frames with one GPU-to-GPU copy.
- **Status:** `/status` includes `vpttrain` and `models.vptft`.

## NitroGen backend (`serve.py --policy nitrogen`, tmux window `nitrogen`, port 8965): gamepad

This serves NVIDIA's open gamepad foundation model NitroGen (github MineDojo/NitroGen, weights `nvidia/NitroGen` → `$DATA/nitrogen/ng.pt`). Install with `uv sync --extra nitrogen`; it needs `transformers<5`, since 5.x renamed SigLIP internals.

- **Replies** are `{"pad": {...}, "version", "infer_ms", "step"}`, with **no** `keys`/`mb`/`cam`; absent fields mean the sidecar leaves keyboard and mouse alone.
- **`pad`** is level state that holds until the next reply: `{"lx","ly","rx","ry"}` in [-1,1] (XInput convention, +y up), `"lt","rt"` in [0,1], and `"buttons"` as a subset of `a b x y lb rb back start ls rs dpad_u dpad_d dpad_l dpad_r`. These are the same names as agentpc-core `Axes`/`Buttons` and `dataset.PAD_*`.
- **Mapping from model output:**
  - stick y is negated, since the model is +down;
  - triggers come from the LEFT/RIGHT_TRIGGER entries of the button vector;
  - buttons are on above 0.5;
  - GUIDE and the RIGHT_* alias buttons are dropped;
  - `start`/`back` are suppressed unless `cfg.allow_menu` is set.
- **Timing:** one 256×256 frame (cv2 INTER_AREA) produces a chunk of 18 actions for a 60 Hz loop. Each connection replays the chunk by wall clock and re-infers on the newest frame when it runs out. That's about 60 ms per inference, every 0.3 s.
- **GPU:** weights stay fp32, with bf16 autocast for compute (bf16 weights break a masked_scatter). It uses about 4 GB on GPU 0.
- **Games:** the released checkpoint carries no game mapping, so it runs unconditionally. `cfg.game` is reserved for later.

Controller labels for fine-tuning: `build_labels` now also returns `pad_axes` (n,6) and `pad_btn` (n,14) from p0.jsonl `axes`/`buttons`.

Note (2026-10-19): the BC `train`/`serve` and base `vpt` tmux windows were stopped to free GPU 0 for NitroGen. Restart them with the commands above if needed.
