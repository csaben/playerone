const { app, BrowserWindow, ipcMain, screen, powerMonitor, globalShortcut, Notification, Tray, Menu, nativeImage, shell } = require('electron');
const fs = require('fs');
const net = require('net');
const path = require('path');
const { execFileSync } = require('child_process');
const { Helper } = require('./helper');
const { Recorder } = require('./recorder');
const { Shipper } = require('./ship');
const { AGENTS, DEFAULT } = require('./agents');

// Installed builds would keep config in %APPDATA%\player-one; a source checkout uses ./config.json.
const CONFIG = app.isPackaged ? path.join(app.getPath('userData'), 'config.json') : path.join(__dirname, '..', 'config.json');
// Only as big as what's showing: a full-size transparent topmost window makes whatever
// is underneath repaint, so collapsed it shrinks to the edge tab.
const SIZES = { tab: [34, 240], full: [560, 660] };
let size = 'tab';
let lastState = '';

let cfg;
let win;
let tray;
let expanded = false;
let pushTimer = null;
let windows = []; // top-level windows, front to back
let prev = ''; // the window that had the keyboard when the panel opened
const attached = new Map(); // hwnd -> { agent, at, status: 'running' | 'done' | 'error', error }
const icons = new Map(); // exe path -> data URL
const helper = new Helper();
const recorder = new Recorder(path.join(app.getPath('userData'), 'rec.log'));
const shipper = new Shipper();
// hwnd -> { title, app, at, status: 'recording' | 'saving' | 'saved' | 'error', dir, error, upload }
const recordings = new Map();
let devices = { connected: 0, enabled: 0x3f };
let datasets = [];

// Where recordings go, as in agentpc: agentpc shards, uploaded to a Hugging Face dataset under
// <window title>-<time>. An empty dataset keeps them local only.
const DEFAULT_DATASET = 'arelius/rlrec-desktop-capture';
const AGENTPC = path.resolve(__dirname, '..', '..', 'agentpc');

function loadConfig() {
  // tabY: where the collapsed tab sits on the edge (0 top, 1 bottom), below openone's
  // recordHotkey: start/stop recording the focused window. devices: bits 0-3 controllers 1-4,
  // 16 keyboard, 32 mouse. server: agentpc-server, for its shared list of datasets.
  // playHotkey: the policy agent starts/stops playing the focused window. policy: the bc server;
  // playModel: which of the trainer's models plays (bc, vpt), each on its own port of that host.
  // playUnfocused / playYield / playResumeMs / playTemp: the Settings view (see PLAY_SETTINGS).
  // recordFps / recordMaxWidth: capture rate and downscale (0 = full size). ingest: the trainer's
  // ingest server (train/ingest.py); when set, each 4 s chunk is sent there as soon as it closes.
  cfg = {
    hotkey: 'Control+Alt+;', recordHotkey: 'Control+Alt+S', playHotkey: "Control+Alt+'", agent: DEFAULT, tabY: 0.72,
    devices: 0x3f, dataset: DEFAULT_DATASET, datasets: [], server: 'http://127.0.0.1:8787',
    recordFps: 20, recordMaxWidth: 960, ingest: '', policy: '', playModel: 'bc',
    playUnfocused: true, playYield: true, playResumeMs: 400, playTemp: 1,
  };
  try { Object.assign(cfg, JSON.parse(fs.readFileSync(CONFIG, 'utf8'))); } catch {}
}

// settings changed in the panel go back into config.json, keeping whatever else is there
function saveConfig(patch) {
  Object.assign(cfg, patch);
  let disk = {};
  try { disk = JSON.parse(fs.readFileSync(CONFIG, 'utf8')); } catch {}
  fs.writeFileSync(CONFIG, `${JSON.stringify({ ...disk, ...patch }, null, 2)}\n`);
}

function display() {
  const all = screen.getAllDisplays();
  return typeof cfg.display === 'number' && all[cfg.display] ? all[cfg.display] : screen.getPrimaryDisplay();
}

function bounds() {
  const a = display().workArea;
  const [w, h] = SIZES[size];
  const at = size === 'tab' ? cfg.tabY : 0.5;
  const y = Math.max(0, Math.min(a.height - h, Math.round(a.height * at - h / 2)));
  return { x: a.x + a.width - w, y: a.y + y, width: w, height: h };
}

function place() {
  if (win && !win.isDestroyed()) win.setBounds(bounds());
}

// Windows moves windows when monitors sleep, wake or reconnect, and display events
// arrive before the layout settles. So place again a little later.
function settle() {
  place();
  setTimeout(place, 1000);
  setTimeout(place, 4000);
}

function createWindow() {
  win = new BrowserWindow({
    width: SIZES.tab[0], height: SIZES.tab[1], frame: false, transparent: true, resizable: false, movable: false,
    // 'toolbar' makes it a Windows tool window: no taskbar button, and not in its own list
    type: 'toolbar', skipTaskbar: true, alwaysOnTop: true, hasShadow: false, show: false, backgroundColor: '#00000000',
    webPreferences: { preload: path.join(__dirname, 'preload.js'), contextIsolation: true, nodeIntegration: false, backgroundThrottling: false },
  });
  win.setAlwaysOnTop(true, 'screen-saver');
  win.setIgnoreMouseEvents(true, { forward: true });
  place();
  win.loadFile(path.join(__dirname, 'ui', 'index.html'));
  win.once('ready-to-show', () => win.showInactive());
  win.on('blur', () => win.webContents.send('collapse'));
  for (const ev of ['display-metrics-changed', 'display-added', 'display-removed']) screen.on(ev, settle);
  for (const ev of ['resume', 'unlock-screen']) powerMonitor.on(ev, settle);
  setInterval(() => { const b = win.getBounds(); const w = bounds(); if (b.x !== w.x || b.y !== w.y || b.width !== w.width) place(); }, 3000);
}

// ---------- windows ----------

async function icon(exe) {
  if (!exe) return '';
  if (!icons.has(exe)) {
    icons.set(exe, '');
    try { icons.set(exe, (await app.getFileIcon(exe, { size: 'normal' })).toDataURL()); } catch {}
  }
  return icons.get(exe);
}

async function refresh() {
  const res = await helper.list(process.pid);
  if (!res.ok) return;
  const list = res.windows || [];
  windows = await Promise.all(list.map(async (w) => ({
    key: String(w.hwnd),
    hwnd: w.hwnd,
    title: w.title,
    app: w.exe ? path.basename(w.exe, path.extname(w.exe)) : w.cls,
    exe: w.exe,
    icon: await icon(w.exe),
  })));
  for (const k of attached.keys()) if (!windows.some((w) => w.key === k)) attached.delete(k);
  push();
}

async function poll() {
  for (;;) {
    await refresh().catch(() => {});
    await new Promise((r) => setTimeout(r, expanded ? 1500 : 5000));
  }
}

function state() {
  return {
    windows: windows.map((w) => ({ ...w, attached: attached.get(w.key) || null, rec: recordings.get(w.key) || null })),
    // recordings whose window is gone (closed while recording, or finished) still show
    recordings: [...recordings.entries()].map(([key, r]) => ({ key, ...r })),
    prev,
    agent: { key: cfg.agent, label: (AGENTS[cfg.agent] || AGENTS[DEFAULT]).label },
    rec: { hotkey: cfg.recordHotkey, dataset: cfg.dataset, datasets, devices },
    play: { hotkey: cfg.playHotkey, model: cfg.playModel, bridge: bridgeUp || [...attached.values()].some((a) => a.bridge),
      unfocused: cfg.playUnfocused, yield: cfg.playYield, resumeMs: cfg.playResumeMs, temp: cfg.playTemp },
    model,
  };
}

function push() {
  if (pushTimer) return;
  pushTimer = setTimeout(() => {
    pushTimer = null;
    const s = JSON.stringify(state());
    if (s !== lastState && win && !win.isDestroyed()) win.webContents.send('state', JSON.parse(s));
    lastState = s;
    const live = [...recordings.values()].filter((r) => r.status === 'recording').length;
    if (tray) tray.setToolTip([live && `${live} recording`, attached.size && `${attached.size} attached`].filter(Boolean).join(' · ') || 'player-one');
  }, 30);
}

// Close the panel first: the helper takes the foreground from the dock, then types.
async function attach(key) {
  const w = windows.find((x) => x.key === key);
  if (!w) return;
  const agent = AGENTS[cfg.agent] || AGENTS[DEFAULT];
  if (agent.policy) return attachPolicy(w);
  const a = { agent: cfg.agent, at: Date.now(), status: 'running', error: '' };
  attached.set(key, a);
  push();
  win.webContents.send('collapse');
  const type = async (text) => {
    const r = await helper.type(w.hwnd, text);
    if (!r.ok) throw new Error(r.error || 'typing failed');
    return r;
  };
  try {
    await agent.run(w, { type });
    a.status = 'done';
  } catch (e) {
    a.status = 'error';
    a.error = String(e.message || e);
    note(`Couldn't drive ${w.app}`, a.error);
  }
  push();
}

function detach(key) {
  const a = attached.get(key);
  attached.delete(key);
  push();
  if (a && a.policy) stopRecording(key);
}

// A policy agent plays inside a recording session, so its play and your corrections are both
// recorded (and streamed to the trainer). A recording already running on the window ends first.
async function attachPolicy(w) {
  const key = w.key;
  if (!cfg.policy) return note("Can't play", 'Set "policy" in config.json to the policy server (ws://host:8962/).');
  const name = cfg.playModel;
  const a = { agent: 'bc', model: name, at: Date.now(), status: 'starting', error: '', policy: true, version: null };
  attached.set(key, a);
  push();
  win.webContents.send('collapse');
  const r = recordings.get(key);
  if (r && r.status === 'recording') await stopRecording(key);
  for (let i = 0; i < 100 && recordings.get(key) && recordings.get(key).status === 'saving'; i++) await new Promise((res) => setTimeout(res, 100));
  await helper.type(w.hwnd, '').catch(() => {});
  // the port comes from the trainer's status, which isn't polled while the panel is closed
  if (cfg.ingest && !(model && model.models)) await fetchModel();
  const ok = await startRecording(key, { policy: playPolicy(name) });
  if (!ok) { attached.delete(key); push(); return; }
  if (attached.get(key) === a && a.status === 'starting') a.status = 'playing';
  push();
}

// ---------- model ----------

// A model's server: the policy host, on the port the trainer's status lists for it.
function policyUrl(name) {
  const port = model && model.models && model.models[name] && model.models[name].port;
  if (!port) return cfg.policy;
  const u = new URL(cfg.policy);
  u.port = String(port);
  return u.toString();
}

// The sidecar's play options, from the Settings view. Temperature 0 is the likeliest action every
// frame (VPT: no camera sampling either); otherwise the server samples the camera at that temperature.
function playPolicy(name) {
  const t = Number(cfg.playTemp);
  return {
    url: policyUrl(name), agent: 1001,
    unfocused: !!cfg.playUnfocused, yield: !!cfg.playYield, resume_ms: Math.max(0, Number(cfg.playResumeMs) || 0),
    cfg: { temperature: t, sample_camera: t > 0 },
  };
}

// Settings view: which keys it may set, and the values each one cycles through.
const PLAY_SETTINGS = {
  playUnfocused: [true, false], playYield: [true, false], playResumeMs: [400, 1000, 3000, 10000], playTemp: [1, 0.5, 0],
};
function setPlay(name, value) {
  const ok = PLAY_SETTINGS[name];
  if (!ok || !ok.includes(value)) return;
  saveConfig({ [name]: value });
  push();
}

// Whether the playerone-bridge mod (mcmod/) is listening: Minecraft with it can play in the
// background. Checked with the model poll, a bare connect and close.
let bridgeUp = false;
function checkBridge() {
  return new Promise((resolve) => {
    const s = net.connect({ host: '127.0.0.1', port: 8970 });
    const done = (up) => { s.destroy(); if (up !== bridgeUp) { bridgeUp = up; push(); } resolve(); };
    s.setTimeout(300, () => done(false));
    s.on('connect', () => done(true));
    s.on('error', () => done(false));
  });
}

// Picks the model the play hotkey uses; windows already playing keep theirs until played again.
function pickModel(name) {
  if (!/^[\w-]+$/.test(name) || name === cfg.playModel) return;
  saveConfig({ playModel: name });
  push();
}

// The trainer's status (train/ingest.py GET /status): how much play it has, how training is
// going, and what the policy server is serving. Polled while the panel is open, or while playing.
let model = null;
async function pollModel() {
  for (;;) {
    const playing = [...attached.values()].some((a) => a.policy);
    if (cfg.ingest && (expanded || playing)) {
      await fetchModel();
      push();
    }
    // not while playing: the sidecar holds the mod's connection then, and reports it itself
    if (expanded && !playing) await checkBridge();
    await new Promise((r) => setTimeout(r, 2000));
  }
}

async function fetchModel() {
  try {
    const res = await fetch(`${cfg.ingest.replace(/\/+$/, '')}/status`, { signal: AbortSignal.timeout(3000) });
    model = res.ok ? await res.json() : { ok: false, error: `status ${res.status}` };
  } catch (e) {
    model = { ok: false, error: e.name === 'TimeoutError' ? 'trainer not answering' : String(e.message || e) };
  }
}

// ---------- recording ----------

// `hf` if it's on PATH, else agentpc's `uv run hf` (where this machine has it), as agentpc does.
// "hfCmd" / "hfCwd" in the config override both.
let hfCache = null;
function hfInvocation() {
  if (cfg.hfCmd) return { hf_cmd: cfg.hfCmd, hf_cwd: cfg.hfCwd || null };
  if (!hfCache) {
    try {
      execFileSync('hf', ['--version'], { stdio: 'ignore', windowsHide: true });
      hfCache = { hf_cmd: ['hf'], hf_cwd: null };
    } catch {
      hfCache = { hf_cmd: ['uv', 'run', 'hf'], hf_cwd: fs.existsSync(AGENTPC) ? AGENTPC : null };
    }
  }
  return hfCache;
}

const stamp = () => {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
};
// agentpc's upload folder: <window title>-<datetime>
const folderFor = (title) => `${(title || 'window').replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'window'}-${stamp()}`;

async function startRecording(key, extra = {}) {
  if (!windows.some((x) => x.key === key)) await refresh().catch(() => {});
  const w = windows.find((x) => x.key === key);
  if (!w) return note("Couldn't record", 'That window is gone.');
  const dataset = String(cfg.dataset || '').trim();
  const ingest = String(cfg.ingest || '').trim();
  const r = { title: w.title, app: w.app, icon: w.icon, at: Date.now(), status: 'recording', dir: '', error: '', upload: dataset, ingest, chunks: 0, shipped: 0 };
  recordings.set(key, r);
  push();
  const res = await recorder.begin({
    hwnd: key, label: w.title, out_dir: path.join(app.getPath('userData'), 'recordings'),
    fps: cfg.recordFps, max_width: cfg.recordMaxWidth, stream: !!ingest, ...extra,
    ...(dataset ? { repo: dataset, folder: folderFor(w.title), ...hfInvocation() } : {}),
  });
  if (!res.ok) {
    recordings.delete(key);
    push();
    note(`Couldn't record ${w.app}`, res.error);
    return false;
  }
  r.dir = res.dir;
  return true;
}

async function stopRecording(key) {
  const r = recordings.get(key);
  if (!r || r.status !== 'recording') return;
  r.status = 'saving';
  r.endedAt = Date.now();
  push();
  const res = await recorder.end(key);
  if (!res.ok) { r.status = 'error'; r.error = res.error; push(); }
}

function toggleRecording(key) {
  const r = recordings.get(key);
  if (r && r.status === 'recording') return stopRecording(key);
  if (r && r.status === 'saving') return null;
  return startRecording(key);
}

// The record and play hotkeys act on the focused window; with the panel open, on the window you were in.
async function targetKey() {
  if (win.isFocused()) return prev;
  const fg = await helper.fg();
  return fg.ok && fg.hwnd ? String(fg.hwnd) : null;
}

async function recordHotkey() {
  const key = await targetKey();
  if (key) toggleRecording(key);
}

async function playHotkey() {
  const key = await targetKey();
  if (key) togglePlay(key);
}

async function togglePlay(key) {
  const a = attached.get(key);
  if (a && a.policy) return detach(key);
  if (!windows.some((x) => x.key === key)) await refresh().catch(() => {});
  const w = windows.find((x) => x.key === key);
  if (w) attachPolicy(w);
}

// play mode: who is driving, which checkpoint, and policy failures
recorder.on('driver', (m) => {
  const a = attached.get(m.hwnd);
  if (a && a.policy && a.status !== 'error') { a.status = m.agent ? 'playing' : 'you'; push(); }
});
recorder.on('play_stats', (m) => {
  const a = attached.get(m.hwnd);
  if (!a || !a.policy) return;
  a.stats = { fps: m.fps, rtt: m.rtt_ms, rttMax: m.rtt_max_ms, infer: m.infer_ms };
  // acting: stats arrive once a second while replies flow, so 2.5 s without one means it stopped
  a.acting = !!m.acting;
  a.bridge = !!m.bridge;
  clearTimeout(actingTimers.get(m.hwnd));
  actingTimers.set(m.hwnd, setTimeout(() => { if (a.acting) { a.acting = false; push(); } }, 2500));
  push();
});
const actingTimers = new Map();
recorder.on('policy', (m) => {
  const a = attached.get(m.hwnd);
  if (a && a.policy) { a.version = m.version; a.step = m.step; push(); }
});
// the model plays with a controller: the sidecar plugged in a virtual Xbox pad (ViGEmBus)
recorder.on('pad', (m) => {
  const a = attached.get(m.hwnd);
  if (!a || !a.policy) return;
  a.pad = !!m.on;
  push();
  if (m.error) note("Can't use a controller", `${m.error}. Install ViGEmBus to let a model play with a controller.`);
});
recorder.on('policy_error', (m) => {
  const a = attached.get(m.hwnd);
  if (!a || !a.policy) return;
  a.status = 'error';
  a.error = m.error;
  push();
  note('Policy stopped', `${m.error}. Still recording; detach to stop.`);
});

recorder.on('finished', (m) => {
  const r = recordings.get(m.hwnd);
  // a session replaced on the same window (the play hotkey during a recording) finishes late: not ours
  if (!r || (m.dir && r.dir && m.dir !== r.dir)) return;
  const p = attached.get(m.hwnd);
  if (p && p.policy) attached.delete(m.hwnd);
  r.status = m.error ? 'error' : 'saved';
  r.error = m.error || '';
  r.dir = m.dir || r.dir;
  r.endedAt = r.endedAt || Date.now();
  push();
  if (m.error) note(`Recording of ${r.app} failed`, m.error);
  else note(`Recording of ${r.app} saved`, [r.upload ? `Uploaded to ${r.upload}` : r.dir, r.ingest && `${r.chunks} chunks to trainer`].filter(Boolean).join(' · '));
  // finished ones stay listed a while, then drop off
  setTimeout(() => { if (recordings.get(m.hwnd) === r) { recordings.delete(m.hwnd); push(); } }, 5 * 60000);
});
// streaming to the trainer: each closed chunk goes out right away
recorder.on('chunk', (m) => {
  const r = recordings.get(m.hwnd);
  if (!r || !r.ingest) return;
  r.chunks++;
  shipper.add(r.ingest, { ...m, rec: r });
  push();
});
shipper.on('sent', (c) => { c.rec.shipped++; c.rec.shipError = ''; push(); });
shipper.on('failed', (c, e) => {
  if (!c.rec.shipError) console.error(`ship ${c.key}: ${e.message || e}`);
  c.rec.shipError = String(e.message || e);
  push();
});
recorder.on('exit', () => {
  for (const r of recordings.values()) if (r.status === 'recording' || r.status === 'saving') { r.status = 'error'; r.error = 'recorder exited'; }
  push();
});

async function loadDevices() {
  const res = await recorder.devices();
  if (res.ok) { devices = { connected: res.connected, enabled: res.enabled }; push(); }
}

async function setDevices(mask) {
  const res = await recorder.setDevices(mask);
  if (!res.ok) return;
  devices.enabled = res.enabled;
  saveConfig({ devices: res.enabled });
  push();
}

const serverUrl = () => String(cfg.server || '').replace(/\/$/, '');

// known datasets: agentpc-server's shared list, plus ones used here before
async function loadDatasets() {
  let shared = [];
  try {
    const r = await fetch(`${serverUrl()}/datasets`, { signal: AbortSignal.timeout(2000) });
    if (r.ok) shared = await r.json();
  } catch {}
  datasets = [...new Set([DEFAULT_DATASET, ...shared, ...(cfg.datasets || []), cfg.dataset].filter(Boolean))];
  push();
}

function setDataset(repo) {
  repo = String(repo || '').trim();
  const isNew = repo && repo !== DEFAULT_DATASET && !(cfg.datasets || []).includes(repo);
  saveConfig({ dataset: repo, ...(isNew ? { datasets: [...(cfg.datasets || []), repo] } : {}) });
  if (isNew) {
    datasets = [...new Set([...datasets, repo])];
    // share it with other machines, best effort, as agentpc does
    fetch(`${serverUrl()}/datasets`, { method: 'POST', body: repo, signal: AbortSignal.timeout(2000) }).catch(() => {});
  }
  push();
}

function note(title, body) {
  new Notification({ title, body: String(body || '').slice(0, 200), silent: true }).show();
}

// ---------- tray ----------

// a small play triangle, drawn here so there are no image assets to keep
function trayIcon() {
  const draw = (n) => {
    const buf = Buffer.alloc(n * n * 4);
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const u = (x + 0.5) / n;
        const v = (y + 0.5) / n;
        const half = (u - 0.2) * 0.6;
        if (u > 0.2 && u < 0.9 && Math.abs(v - 0.5) < half) buf.writeUInt32LE(0xffffffff, (y * n + x) * 4);
      }
    }
    return nativeImage.createFromBitmap(buf, { width: n, height: n });
  };
  const img = draw(16);
  img.addRepresentation({ scaleFactor: 2, width: 32, height: 32, buffer: draw(32).toBitmap() });
  return img;
}

function menuTemplate() {
  return [
    { label: `Open panel  (${cfg.hotkey.replace('Control', 'Ctrl')})`, click: toggle },
    { type: 'separator' },
    ...Object.entries(AGENTS).map(([k, ag]) => ({
      label: `Agent: ${ag.label}`, type: 'radio', checked: cfg.agent === k, click: () => { cfg.agent = k; push(); },
    })),
    { type: 'separator' },
    { label: `Record focused window  (${cfg.recordHotkey.replace('Control', 'Ctrl')})`, click: recordHotkey },
    { label: 'Open recordings folder', click: () => { const d = path.join(app.getPath('userData'), 'recordings'); fs.mkdirSync(d, { recursive: true }); shell.openPath(d); } },
    { type: 'separator' },
    { label: 'Detach all', enabled: attached.size > 0, click: () => { attached.clear(); push(); } },
    { label: 'Reload config', click: () => { loadConfig(); place(); registerHotkey(); } },
    { label: 'Open config', click: () => { if (!fs.existsSync(CONFIG)) fs.writeFileSync(CONFIG, `${JSON.stringify({ hotkey: cfg.hotkey }, null, 2)}\n`); shell.openPath(CONFIG); } },
    { type: 'separator' },
    { label: 'Quit player-one', click: () => app.quit() },
  ];
}

async function toggle() {
  if (win.isFocused()) { win.webContents.send('collapse'); win.blur(); return; }
  // ask before taking the keyboard, so the window you were in can be first in the list
  const fg = await helper.fg();
  if (fg.ok && fg.hwnd) { prev = String(fg.hwnd); push(); }
  if (size !== 'full') { size = 'full'; place(); }
  win.show();
  win.focus();
  win.webContents.focus();
  win.webContents.send('expand', prev);
  refresh().catch(() => {});
}

function registerHotkey() {
  globalShortcut.unregisterAll();
  for (const [name, key, fn] of [['hotkey', cfg.hotkey, toggle], ['recordHotkey', cfg.recordHotkey, recordHotkey], ['playHotkey', cfg.playHotkey, playHotkey]]) {
    if (!key) continue;
    let ok = false;
    try { ok = globalShortcut.register(key, fn); } catch {}
    if (!ok) note('player-one hotkey not set', `${key} is taken by another app or not a valid key. Change "${name}" in the config.`);
  }
}

// Chromium treats a mostly-transparent topmost window as occluded and stops painting it.
app.commandLine.appendSwitch('disable-features', 'CalculateNativeWinOcclusion');

if (!app.requestSingleInstanceLock()) app.quit();

app.whenReady().then(() => {
  app.setAppUserModelId('com.csaben.player-one');
  loadConfig();
  createWindow();
  helper.start();
  tray = new Tray(trayIcon());
  tray.setToolTip('player-one');
  tray.on('click', toggle);
  tray.on('right-click', () => tray.popUpContextMenu(Menu.buildFromTemplate(menuTemplate())));
  registerHotkey();
  poll();
  pollModel();
  if (recorder.start()) recorder.setDevices(cfg.devices).then(loadDevices);
  loadDatasets();
  if (process.env.PO_EXPAND) setTimeout(toggle, 3000);

  ipcMain.on('ui:size', (_e, s) => { if (SIZES[s] && s !== size) { size = s; place(); } });
  ipcMain.on('ui:ignore', (_e, ignore) => win.setIgnoreMouseEvents(!!ignore, { forward: true }));
  ipcMain.on('ui:focus', () => { win.focus(); win.webContents.focus(); });
  ipcMain.on('ui:menu', () => Menu.buildFromTemplate(menuTemplate()).popup({ window: win }));
  ipcMain.on('ui:ready', () => { lastState = ''; push(); });
  ipcMain.on('ui:expanded', (_e, v) => { expanded = !!v; });
  ipcMain.handle('win:attach', (_e, key) => attach(String(key)));
  ipcMain.handle('win:detach', (_e, key) => detach(String(key)));
  ipcMain.handle('win:play', (_e, key) => togglePlay(String(key)));
  ipcMain.handle('model:pick', (_e, name) => pickModel(String(name)));
  ipcMain.handle('play:set', (_e, name, value) => setPlay(String(name), value));
  ipcMain.handle('rec:toggle', (_e, key) => toggleRecording(String(key)));
  ipcMain.handle('rec:devices', () => loadDevices());
  ipcMain.handle('rec:set-devices', (_e, mask) => setDevices(Number(mask) & 0x3f));
  ipcMain.handle('rec:datasets', () => loadDatasets());
  ipcMain.handle('rec:set-dataset', (_e, repo) => setDataset(repo));
  ipcMain.handle('rec:open', (_e, dir) => { if (dir && fs.existsSync(dir)) shell.openPath(dir); });
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
  helper.stop();
  recorder.stop(); // it finalizes any running recordings before exiting
});
app.on('window-all-closed', () => app.quit());
