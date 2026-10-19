# player-one

A Windows edge dock, styled after [openone](../openone), for attaching an AI to any window.

**Ctrl+Alt+B** opens the panel on a list of your open windows, with the one you were just in selected. **j**/**k** (or arrows) move, **/** fuzzy-searches by app or title (arrows or Ctrl+j/k pick a match), **Enter** attaches, **x** detaches, **Esc** closes.

Attaching closes the panel, brings that window forward and runs the current agent on it. For now the only agent is a macro that types `hello world` (`src/agents.js`); a real AI goes in there. Typing stops if the window loses the foreground, so keys never land somewhere else. Windows running as admin can't receive keys from a non-admin player-one.

## Recording

**Ctrl+Alt+S** starts recording the focused window, and pressing it again in that window stops it. With the panel open, **r** records the selected window. Recording works like agentpc's human demonstration mode (../agentpc): the window's frames (WGC, about 5 fps) plus the keyboard, mouse and Xbox controller input you give **while that window is focused**, written as agentpc shards (`agentpc.trajectory/v3` events, mp4 and actions per 4 s chunk; see agentpc's `docs/SCHEMA.md`) to `%APPDATA%player-oneecordings`.

**Tab** in the panel opens the record card:

- **Recordings**: running ones (Enter stops) and finished ones (Enter opens the folder).
- **Devices**: controllers 1-4 (shows which are connected), keyboard, mouse. A device that is off is never recorded.
- **Upload to**: where finished recordings go. As in agentpc, that's a Hugging Face dataset (default `arelius/rlrec-desktop-capture`, private), each recording in its own `<window title>-<time>` folder, uploaded with `hf upload` (or `uv run hf` from ../agentpc when `hf` isn't on PATH). The list also includes agentpc-server's shared datasets (`server`, default http://127.0.0.1:8787) when it's running. **Local only** skips the upload.

The recorder is `native/`: a Rust sidecar (`player-one-rec`) built on agentpc's `core`, `capture` and `record` crates, copied into `native/crates` (see its README for the one change). Build it once:

```powershell
cd native; cargo build --release
```

## Run

```powershell
npm install
npm start            # or .\start-player-one.cmd
npm test
```

Optional `config.json` next to `package.json` (tray > Open config): `{ "hotkey": "Control+Alt+B", "recordHotkey": "Control+Alt+S", "tabY": 0.72, "display": 0 }`. The record card saves `devices` and `dataset` there too; `hfCmd` / `hfCwd` override how `hf` is run. `tabY` is where the collapsed tab sits on the right edge (0 top, 1 bottom), below openone's by default.

## Layout

- `src/main.js`: dock window, tray, hotkey, attach flow
- `src/win-helper.ps1` + `src/helper.js`: one warm PowerShell that lists windows (EnumWindows, Alt+Tab rules), reports the foreground window, and focuses + types (SendInput, unicode)
- `src/agents.js`: what runs on attach
- `src/recorder.js` + `native/rec`: the recorder sidecar (JSON lines, like the helper)
- `src/ui/`: the panel (openone's look, trimmed)

## Demo

`docs/demo.mp4`, viewable on the tailnet at https://inari.tailb1b51d.ts.net/player-one/ (`node tools/serve-docs.js` on :8841, proxied by `tailscale serve --bg --set-path /player-one http://127.0.0.1:8841`).
