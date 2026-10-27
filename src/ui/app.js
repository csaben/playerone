const $ = (s) => document.querySelector(s);
const root = $('#root');
const body = $('#card-body');

let S = { windows: [], recordings: [], agent: { key: 'hello', label: 'hello world' }, rec: { hotkey: '', dataset: '', datasets: [], devices: { connected: 0, enabled: 0x3f } } };
let expanded = false;
let view = 'windows'; // 'windows' | 'rec'
let kb = -1;
let sig = '';
let hintTimer = null;
let query = null; // the "/" search: null when closed
let wantPrev = false; // just opened: select the window you were in once main says which

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const X = '<svg viewBox="0 0 16 16"><path d="m4 4 8 8M12 4l-8 8"/></svg>';
const WIN = '<svg viewBox="0 0 16 16"><rect x="2" y="3" width="12" height="10" rx="2"/><path d="M2 6h12"/></svg>';
const PAD = '<svg viewBox="0 0 16 16"><path d="M4.5 4.5h7a3 3 0 0 1 3 3.2l-.3 2.6a1.8 1.8 0 0 1-3.2.9L10 10H6l-1 1.2a1.8 1.8 0 0 1-3.2-.9l-.3-2.6a3 3 0 0 1 3-3.2Z"/><path d="M5 7v2M4 8h2"/><circle cx="10.6" cy="7.4" r=".6"/><circle cx="11.8" cy="8.6" r=".6"/></svg>';
const KEYS = '<svg viewBox="0 0 16 16"><rect x="1.5" y="4" width="13" height="8" rx="1.5"/><path d="M4 6.5h.01M6.5 6.5h.01M9 6.5h.01M11.5 6.5h.01M5 9.5h6"/></svg>';
const MOUSE = '<svg viewBox="0 0 16 16"><rect x="4.5" y="2" width="7" height="12" rx="3.5"/><path d="M8 4.5v2"/></svg>';
const CLOUD = '<svg viewBox="0 0 16 16"><path d="M4.5 12.5h7a3 3 0 0 0 .4-6A4 4 0 0 0 4.2 7a2.8 2.8 0 0 0 .3 5.5Z"/></svg>';
const DISK = '<svg viewBox="0 0 16 16"><path d="M3 3h8l2 2v8H3Z"/><path d="M5.5 3v3h5V3M5.5 13v-3.5h5V13"/></svg>';

// rail marks, as in openone: the blue spinner while an agent acts, a grey dot while it waits
const status = (a) => (a.status === 'running' || a.acting ? 'working'
  : a.status === 'error' || a.status === 'done' ? a.status : 'idle');
const attachedList = () => S.windows.filter((w) => w.attached);
const live = () => (S.recordings || []).filter((r) => r.status === 'recording');
// attached first (most recent attach on top), then the window you were just in, then the
// rest front to back
const ordered = () => {
  const rest = S.windows.filter((w) => !w.attached);
  const i = rest.findIndex((w) => w.key === S.prev);
  if (i > 0) rest.unshift(...rest.splice(i, 1));
  return [...attachedList().sort((a, b) => b.attached.at - a.attached.at), ...rest];
};
const find = (key) => S.windows.find((w) => w.key === key);
const keyLabel = (k) => String(k || '').replace('Control', 'Ctrl').split('+').map((x) => `<kbd>${esc(x)}</kbd>`).join('');

function ago(t) {
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return 'now';
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  return `${Math.floor(s / 86400)}d`;
}

// 0:07, 12:40, 1:02:03
function clock(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  const p = (n) => String(n).padStart(2, '0');
  return s >= 3600 ? `${Math.floor(s / 3600)}:${p(Math.floor(s / 60) % 60)}:${p(s % 60)}` : `${Math.floor(s / 60)}:${p(s % 60)}`;
}

function marks(max) {
  // a playing model records its window too; its agent mark stands for both
  const rec = live().filter((r) => !((find(r.key) || {}).attached || {}).policy);
  const all = [...rec.map(() => 'rec'), ...attachedList().map((w) => status(w.attached))];
  const shown = all.slice(0, max).map((st) => `<span class="st ${st}"></span>`).join('');
  const more = all.length > max ? `<span class="more">+${all.length - max}</span>` : '';
  return shown + more;
}

// ---------- render ----------

function setHtml(el, html) {
  if (el.dataset.html !== html) { el.innerHTML = html; el.dataset.html = html; }
}

let shrinkTimer = null;
function fit() {
  clearTimeout(shrinkTimer);
  if (expanded) return window.po.size('full');
  // let the close transition finish before shrinking the window
  shrinkTimer = setTimeout(() => { if (!expanded) window.po.size('tab'); }, 260);
}

function renderChrome() {
  setHtml($('#tab-marks'), marks(7));
  setHtml($('#b-attached'), marks(6) || '<span class="st idle"></span>');
  $('#b-wins').classList.toggle('on', expanded && view === 'windows');
  $('#b-rec').classList.toggle('on', expanded && view === 'rec');
  $('#b-set').classList.toggle('on', expanded && view === 'settings');
  const P = S.play || {};
  $('#b-set').title = `Play settings · ${P.unfocused && P.bridge ? 'plays in the background' : 'plays while focused'} · ${P.yield ? 'pauses while you play' : 'shared control'}`;
  $('#b-model').classList.toggle('on', expanded && view === 'model');
  $('#b-rec').classList.toggle('live', live().length > 0);
}

function chip(a) {
  const step = a.step != null ? ` · step ${a.step}` : '';
  const text = a.status === 'running' ? `${S.agent.label} · typing` : a.status === 'error' ? 'failed'
    : a.status === 'playing' ? `playing${a.model && a.model !== 'bc' ? ` · ${a.model}` : step}` : a.status === 'you' ? 'you have control' : a.status === 'starting' ? 'starting…' : 'attached';
  return `<span class="chip ai ${a.status}" title="${esc(a.error || '')}">${esc(text)}</span>`;
}

function recChip(r) {
  if (r.status === 'recording') return `<span class="chip rec"><i></i>${clock(Date.now() - r.at)}</span>`;
  if (r.status === 'saving') return `<span class="chip ai">${r.upload ? 'uploading…' : 'saving…'}</span>`;
  if (r.status === 'error') return `<span class="chip ai error" title="${esc(r.error)}">rec failed</span>`;
  return '<span class="chip ai done">saved</span>';
}

function windowsCard() {
  let i = 0;
  const row = (w) => {
    const a = w.attached;
    const r = w.rec;
    return `<div class="row ${a ? 'on' : ''} ${i++ === kb ? 'kb' : ''}" data-attach="${esc(w.key)}">
      <span class="ico">${w.icon ? `<img src="${w.icon}" alt="">` : WIN}</span>
      <div class="mid"><div class="t1"><span class="nm">${esc(w.title)}</span></div>
      <div class="t2">${esc(w.app)}${a && a.error ? ` · ${esc(a.error)}` : ''}</div></div>
      ${r ? recChip(r) : ''}
      ${a ? `${chip(a)}<span class="age">${esc(ago(a.at))}</span><button class="icon" data-detach="${esc(w.key)}" title="Detach">${X}</button>` : ''}</div>`;
  };
  const all = ordered();
  let list;
  if (query !== null) {
    const found = window.POSearch.filter(query, all);
    list = `<div class="search"><input id="q" placeholder="Search windows: app or title" spellcheck="false" autocomplete="off" value="${esc(query)}"></div>
      ${found.map(row).join('') || '<div class="empty">No matches</div>'}`;
  } else {
    const on = all.filter((w) => w.attached);
    const off = all.filter((w) => !w.attached);
    list = (on.length ? `<div class="host"><span class="dot"></span>Attached<span class="count">${on.length}</span></div>${on.map(row).join('')}` : '')
      + `<div class="host"><span class="dot" style="background:var(--dim)"></span>Windows<span class="count">${off.length}</span></div>`
      + (off.map(row).join('') || '<div class="empty">No windows</div>');
  }
  const n = attachedList().length;
  const html = `<div class="head"><h1>Attach <span style="color:var(--muted);font-weight:500;font-size:12.5px">· ${esc(S.agent.label)}${n ? ` · ${n} attached` : ''}</span></h1><button class="x" data-close>${X}</button></div>${list}`;
  const hint = query !== null
    ? '<kbd>↑</kbd><kbd>↓</kbd> pick · <kbd>Enter</kbd> attach · <kbd>Esc</kbd> clear'
    : `<kbd>Enter</kbd> attach · <kbd>b</kbd> play · <kbd>r</kbd> record · <kbd>/</kbd> search${n ? ' · <kbd>x</kbd> detach' : ''} · <kbd>h</kbd><kbd>l</kbd> views`;
  return { html, hint };
}

const pct = (v) => (v == null ? '–' : `${Math.round(v * 100)}%`);
const since = (t) => (t ? `${ago(t * (t < 1e12 ? 1000 : 1))}` : '–');

// The models view: each model (for now the one behavior clone the trainer serves) with its
// status and latency, then the pipeline behind it and the windows it is playing.
function modelCard() {
  const m = S.model || { ok: false, error: 'checking…' };
  const playing = (S.windows || []).filter((w) => w.attached && w.attached.policy);
  const html = `<div class="head"><h1>Models <span style="color:var(--muted);font-weight:500;font-size:12.5px">· ${keyLabel(S.play && S.play.hotkey)} plays the focused window</span></h1><button class="x" data-close>${X}</button></div>
    ${modelRows(m, playing)}`;
  const hint = '<kbd>j</kbd><kbd>k</kbd> <kbd>Enter</kbd> pick the model that plays · <kbd>h</kbd><kbd>l</kbd> views · <kbd>Esc</kbd> close';
  return { html, hint };
}

// VPT fine-tuned on your play: its step, and how it does on new play once some has come in (the
// trainer's first score is the base model's, at step 0)
function vptftLine(ms, vt) {
  const f = vt.fresh || {};
  const scored = f.at_step > 0;
  const sub = scored ? `new play cam ${pct(f.cam_move_acc)} / keys ${pct(f.keys_recall)}`
    : `${/cap/.test(vt.state || '') ? 'tuned, waiting for more play' : esc((vt.state || 'tuning').split(':')[0])}`;
  return [`VPT tuned${ms.step != null ? ` v${ms.step}` : ''}`, `on your play · ${sub}`];
}

// NitroGen fine-tuned on your controller play: its step, and how its sticks do on new play (the
// trainer's first score is the base model's, at step 0)
function ngftLine(ms, nt) {
  const f = nt.fresh || {};
  const mae = (v) => (v == null ? '–' : Number(v).toFixed(2));
  const sub = f.at_step > 0 ? `new play stick err ${mae(f.stick_mae)} / buttons ${pct(f.btn_recall)}`
    : `${/cap/.test(nt.state || '') ? 'tuned, waiting for more play' : esc((nt.state || 'tuning').split(':')[0])}`;
  return [`NitroGen tuned${ms.step ? ` v${ms.step}` : ''}`, `on your controller play · ${sub}`];
}

function modelRows(m, playing) {
  const stat = (ico, name, sub, st) => `<div class="row stat"><span class="ico">${ico}</span>
      <div class="mid"><div class="t1"><span class="nm">${name}</span></div><div class="t2">${sub}</div></div>
      ${st ? `<span class="chip ai ${st[1]}">${esc(st[0])}</span>` : ''}</div>`;
  if (!m.ok) return stat(CLOUD, 'behavior clone', esc(m.error || 'trainer unreachable'), ['offline', 'error']);
  const ing = m.ingest || {};
  const tr = m.train || {};
  const sv = m.serve || {};
  const fresh = tr.fresh || {};
  const met = Object.keys(fresh).length ? fresh : tr.train || {};
  const tone = (s) => (s === 'down' ? 'error' : s === 'training' || s === 'serving' ? 'done' : 'starting');
  const play = `${(ing.minutes || 0).toFixed(1)} min of play · ${ing.chunks || 0} chunks · last ${since(ing.last_chunk_at)}`;
  const train = tr.step != null ? `step ${tr.step} · ${Math.round(tr.epochs || 0)} passes over the play${/cap/.test(tr.state || '') ? ' (cap: needs more play)' : ''}` : 'no data yet';
  // "fresh" = scored on chunks before training on them, the honest number; else training accuracy
  const acc = `camera moves ${pct(met.cam_move_acc)} · keys ${pct(met.keys_recall)} · clicks ${pct(met.mb_acc)}`;
  const client = (sv.clients || [])[0];
  const serve = sv.step != null
    ? `step ${sv.step} · loaded ${since(sv.loaded_at)}${client ? ` · ${Math.round(client.fps || 0)} fps to you` : ''}`
    : 'no model yet';
  // one line per model: what it is, how good on new play, how fast while playing; the picked one
  // is what the play hotkey starts
  const picked = (S.play && S.play.model) || 'bc';
  const models = m.models || { bc: sv };
  let i = 0;
  // a model whose server is down (stopped on cradle) is hidden unless it's the picked one
  const top = Object.entries(models).filter(([name, ms]) => ms.state !== 'down' || name === picked).map(([name, ms]) => {
    const st = (playing.find((w) => w.attached.stats && (w.attached.model || 'bc') === name) || { attached: {} }).attached.stats;
    const down = name === 'bc' ? [['recording in', ing.state], ['trainer', tr.state], ['server', ms.state]] : [['server', ms.state]];
    const bad = down.filter(([, s]) => s === 'down').map(([n]) => n);
    const what = name === 'bc'
      ? [`behavior clone${ms.step != null ? ` v${ms.step}` : ''}`, `${(ing.minutes || 0).toFixed(1)} min data · new play cam ${pct(fresh.cam_move_acc)} / keys ${pct(fresh.keys_recall)}`]
      : name === 'vpt' ? [`VPT <span style="color:var(--muted);font-weight:500">${esc(String(ms.version || '').replace(/^vpt:/, '').replace(/-2x$/, ''))}</span>`, 'OpenAI pretrained']
      : name === 'vptft' ? vptftLine(ms, m.vpttrain || {})
      : name === 'nitrogen' ? ['NitroGen', 'NVIDIA open gamepad model · plays with a virtual controller']
      : name === 'ngft' ? ngftLine(ms, m.ngtrain || {})
      : [esc(name), esc(ms.version || '')];
    const line = [what[1], st ? `rtt ${Math.round(st.rtt)} ms · ${Math.round(st.fps)} fps` : ms.state === 'serving' ? 'ready' : ms.state || '?'].join(' · ');
    const chipTxt = bad.length ? [`${bad.join(', ')} down`, 'error'] : [ms.state === 'serving' ? 'serving' : ms.state || '?', tone(ms.state)];
    return `<div class="row ${i++ === kb ? 'kb' : ''}" data-model="${esc(name)}"><span class="ico">${PAD}</span>
      <div class="mid"><div class="t1"><span class="nm">${what[0]}</span></div><div class="t2">${line}</div></div>
      <span class="chip ai ${chipTxt[1]}">${esc(chipTxt[0])}</span><span class="radio ${name === picked ? 'on' : ''}"></span></div>`;
  }).join('');
  const plays = playing.map((w) => {
    const a = w.attached;
    const s = a.stats;
    const sub = s ? `${a.bridge ? 'via mod (plays in background) · ' : ''}${a.pad ? 'virtual controller · ' : ''}rtt ${Math.round(s.rtt)} ms · ${Math.round(s.fps)} fps · model ${(s.infer || 0).toFixed(1)} ms` : esc(a.error || 'connecting…');
    return `<div class="row stat"><span class="ico">${w.icon ? `<img src="${w.icon}" alt="">` : WIN}</span>
      <div class="mid"><div class="t1"><span class="nm">${esc(w.title)}</span></div><div class="t2">${sub}</div></div>${chip(a)}</div>`;
  }).join('');
  // the BC pipeline only while it's running or picked; recordings feed every trainer either way
  const bcShown = picked === 'bc' || tr.state !== 'down';
  return top
    + (plays ? `<div class="host"><span class="dot" style="background:#3fcf74"></span>Playing<span class="count">${playing.length}</span></div>${plays}` : '')
    + (!bcShown ? stat(DISK, 'Recorded', play, null) : '<div class="host"><span class="dot" style="background:var(--dim)"></span>Behavior clone pipeline</div>'
    + stat(DISK, 'Recorded', play, null)
    + stat(PAD, 'Training', train, [(tr.state || '?').split(':')[0], tone(tr.state)])
    + (tr.step != null ? stat(KEYS, Object.keys(fresh).length ? 'On new play' : 'On seen play', acc, null) : '')
    + stat(CLOUD, 'Serving', serve, [sv.state || '?', tone(sv.state)]));
}

// The settings view: how a model plays. Each row cycles through the values main.js allows
// (PLAY_SETTINGS); they apply the next time a window starts playing.
const PLAY_VALUES = { playUnfocused: [true, false], playYield: [true, false], playResumeMs: [400, 1000, 3000, 10000], playTemp: [1, 0.5, 0] };
const SET_KEY = { playUnfocused: 'unfocused', playYield: 'yield', playResumeMs: 'resumeMs', playTemp: 'temp' };
function settingsCard() {
  const P = S.play || {};
  let i = 0;
  const row = (key, ico, name, sub, right, dis) => `<div class="row ${i++ === kb ? 'kb' : ''} ${dis ? 'dis' : ''}" data-set="${key}">
      <span class="ico">${ico}</span>
      <div class="mid"><div class="t1"><span class="nm">${name}</span></div><div class="t2">${sub}</div></div>${right}</div>`;
  const check = (on) => `<span class="check ${on ? 'on' : ''}"></span>`;
  const val = (t) => `<span class="val">${t}</span>`;
  const temp = { 1: ['Normal', 'Samples the camera like the model learned it'], 0.5: ['Low', 'Steadier: sharper camera choices'], 0: ['Off', 'The likeliest action every frame'] }[P.temp] || [String(P.temp), ''];
  const html = `<div class="head"><h1>Settings <span style="color:var(--muted);font-weight:500;font-size:12.5px">· how a model plays</span></h1><button class="x" data-close>${X}</button></div>
    ${row('playUnfocused', WIN, 'Plays when out of focus', P.bridge ? 'Keeps playing Minecraft while you use other windows' : 'Needs the player-one Minecraft mod (not running)', check(P.unfocused && P.bridge), !P.bridge)}
    ${row('playYield', KEYS, 'Pauses while you play', P.yield ? 'Your input takes over; it picks up after you stop' : 'Shared control: it keeps acting while you play', check(P.yield))}
    ${row('playResumeMs', PAD, 'Picks up after', 'How long you stop before it takes over again', val(`${(P.resumeMs || 0) / 1000} s`), !P.yield)}
    ${row('playTemp', CLOUD, 'Randomness', temp[1], val(temp[0]))}`;
  const hint = '<kbd>j</kbd><kbd>k</kbd> pick · <kbd>Space</kbd> change · applies on next play · <kbd>h</kbd><kbd>l</kbd> views';
  return { html, hint };
}

// The record card: what's recording, which devices are recorded, and where recordings go.
function recCard() {
  const R = S.rec || {};
  const dev = R.devices || { connected: 0, enabled: 0x3f };
  let i = 0;
  const sel = () => (i++ === kb ? 'kb' : '');
  const check = (on) => `<span class="check ${on ? 'on' : ''}"></span>`;
  const radio = (on) => `<span class="radio ${on ? 'on' : ''}"></span>`;

  const recs = (S.recordings || []).slice().sort((a, b) => b.at - a.at);
  const recRows = recs.map((r) => {
    const act = r.status === 'recording' ? `data-recstop="${esc(r.key)}"` : r.dir ? `data-open="${esc(r.dir)}"` : '';
    const sub = r.status === 'recording' ? `${esc(r.app)} · ${r.upload ? `to ${esc(r.upload)}` : 'local only'}`
      : r.status === 'error' ? esc(r.error) : `${esc(r.app)} · ${clock((r.endedAt || Date.now()) - r.at)}${r.status === 'saved' ? (r.upload ? ` · uploaded to ${esc(r.upload)}` : ' · saved') : ''}`;
    // streaming to the trainer: chunks it has, and why the rest are waiting
    const ship = r.ingest ? ` · trainer ${r.shipped}/${r.chunks}${r.shipError ? ` (retrying: ${esc(r.shipError)})` : ''}` : '';
    return `<div class="row ${sel()}" ${act}>
      <span class="ico">${r.icon ? `<img src="${r.icon}" alt="">` : WIN}</span>
      <div class="mid"><div class="t1"><span class="nm">${esc(r.title)}</span></div><div class="t2">${sub}${ship}</div></div>
      ${recChip(r)}</div>`;
  }).join('');

  const devRow = (bit, ico, name, sub) => `<div class="row ${sel()}" data-dev="${bit}">
      <span class="ico">${ico}</span>
      <div class="mid"><div class="t1"><span class="nm">${name}</span></div><div class="t2">${sub}</div></div>
      ${check(dev.enabled & bit)}</div>`;
  const pads = [0, 1, 2, 3].map((n) => devRow(1 << n, PAD, `Controller ${n + 1}`, dev.connected & (1 << n) ? 'Connected' : 'Not connected')).join('');

  const dsRow = (repo) => `<div class="row ${sel()}" data-ds="${esc(repo)}">
      <span class="ico">${repo ? CLOUD : DISK}</span>
      <div class="mid"><div class="t1"><span class="nm">${repo ? esc(repo) : 'Local only'}</span></div>
      <div class="t2">${repo ? 'Hugging Face dataset, a folder per recording' : 'Keep recordings on this PC'}</div></div>
      ${radio((R.dataset || '') === repo)}</div>`;
  const sets = [...(R.datasets || []), ''].map(dsRow).join('');

  const html = `<div class="head"><h1>Record <span style="color:var(--muted);font-weight:500;font-size:12.5px">· ${keyLabel(R.hotkey)} on any window</span></h1><button class="x" data-close>${X}</button></div>
    ${recs.length ? `<div class="host"><span class="dot" style="background:#e5534b"></span>Recordings<span class="count">${recs.length}</span></div>${recRows}` : ''}
    <div class="host"><span class="dot" style="background:var(--dim)"></span>Devices</div>
    ${pads}
    ${devRow(16, KEYS, 'Keyboard', 'Every keyboard')}
    ${devRow(32, MOUSE, 'Mouse', 'Buttons, movement and wheel')}
    <div class="host"><span class="dot" style="background:var(--dim)"></span>Upload to</div>
    ${sets}
    <div class="row ${sel()}" data-dsnew><span class="ico">${CLOUD}</span><div class="mid"><input id="ds" placeholder="Other dataset: user/name" spellcheck="false" autocomplete="off"></div></div>`;
  const hint = '<kbd>j</kbd><kbd>k</kbd> pick · <kbd>Space</kbd> toggle · <kbd>h</kbd><kbd>l</kbd> views · <kbd>Esc</kbd> close';
  return { html, hint };
}

function render() {
  renderChrome();
  if (!expanded) return;
  const { html, hint } = view === 'rec' ? recCard() : view === 'model' ? modelCard() : view === 'settings' ? settingsCard() : windowsCard();
  const next = `${view}|${html}|${hint}`;
  if (next !== sig) {
    // keep the search or dataset box focused, caret and all, across the rebuild
    const a = document.activeElement;
    const focused = a && (a.id === 'q' || a.id === 'ds') ? a.id : null;
    const caret = focused ? a.selectionStart : 0;
    const value = focused === 'ds' ? a.value : null;
    const top = body.scrollTop;
    body.innerHTML = html;
    body.scrollTop = top;
    if ($('#hint').dataset.text !== hint) { $('#hint').innerHTML = `<span>${hint}</span>`; $('#hint').dataset.text = hint; showHint(!!hint); }
    sig = next;
    const el = focused && document.getElementById(focused);
    if (el) { if (value !== null) el.value = value; el.focus(); el.setSelectionRange(caret, caret); }
  }
  layout();
}

// The key hint is a reminder, not a control: show it for a few seconds when the card
// opens or its keys change, then get out of the way.
function showHint(on) {
  clearTimeout(hintTimer);
  $('#hint').classList.toggle('show', on);
  if (on) hintTimer = setTimeout(() => $('#hint').classList.remove('show'), 4000);
}

function layout() {
  const card = $('#card').getBoundingClientRect();
  const btn = (view === 'rec' ? $('#b-rec') : $('#b-wins')).getBoundingClientRect();
  const y = btn.top + btn.height / 2 - card.top - 6;
  $('.caret').style.top = `${Math.max(16, Math.min(card.height - 28, y))}px`;
  $('#hint').style.top = `${card.bottom + 12}px`;
}

// ---------- actions ----------

function expand(prev) {
  if (prev) S.prev = prev;
  expanded = true;
  view = 'windows';
  window.po.expanded(true);
  window.po.size('full');
  kb = 0;
  wantPrev = true;
  selectPrev();
  void root.offsetWidth;
  root.classList.replace('collapsed', 'expanded');
  sig = '';
  $('#hint').dataset.text = '';
  render();
  window.po.focus();
}

function collapse() {
  if (!expanded) return;
  expanded = false;
  window.po.expanded(false);
  query = null;
  root.classList.replace('expanded', 'collapsed');
  showHint(false);
  fit();
  if (document.activeElement) document.activeElement.blur();
  render();
}

function show(v) {
  view = v;
  query = null;
  kb = 0;
  sig = '';
  if (document.activeElement) document.activeElement.blur();
  if (v === 'rec') { window.po.devices(); window.po.datasets(); } else selectPrev();
  render();
  body.scrollTop = 0;
}

// the window you were just in, so Enter attaches to it
function selectPrev() {
  const i = ordered().findIndex((w) => w.key === S.prev);
  if (i < 0) return;
  wantPrev = false;
  kb = i;
  render();
}

const rows = () => [...body.querySelectorAll('.row:not(.stat)')];
const selected = () => rows()[Math.max(kb, 0)];

function attach(key) {
  if (!key) return;
  collapse();
  window.po.attach(key).catch((err) => toast(err.message));
}

function detach(key) {
  if (!key || !(find(key) || {}).attached) return;
  window.po.detach(key);
}

function play(key) {
  if (!key) return;
  collapse();
  window.po.play(key).catch((err) => toast(err.message));
}

function record(key) {
  if (key) window.po.record(key).catch((err) => toast(err.message));
}

// a row of the record card: stop a recording, open a finished one, flip a device, pick a dataset
function activate(t) {
  if (!t) return;
  const d = t.dataset;
  if (d.recstop) record(d.recstop);
  else if (d.open) window.po.open(d.open);
  else if (d.dev) window.po.setDevices((S.rec.devices.enabled ^ Number(d.dev)) & 0x3f);
  else if (d.ds !== undefined) window.po.setDataset(d.ds);
  else if (d.dsnew !== undefined) { const i = $('#ds'); if (i) i.focus(); }
  else if (d.model) window.po.pickModel(d.model);
  else if (d.set && !t.classList.contains('dis')) {
    const vals = PLAY_VALUES[d.set];
    window.po.setPlay(d.set, vals[(vals.indexOf(S.play[SET_KEY[d.set]]) + 1) % vals.length]);
  }
}

function step(d) {
  const n = rows().length;
  if (!n) return;
  kb = kb < 0 ? (d > 0 ? 0 : n - 1) : (kb + d + n) % n;
  render();
  const row = body.querySelector('.row.kb');
  if (row) row.scrollIntoView({ block: 'nearest' });
}

// ---------- wiring ----------

let ignoring = true;
window.addEventListener('mousemove', (e) => {
  const hit = !!(e.target.closest && e.target.closest('.hit'));
  if (hit === ignoring) {
    ignoring = !hit;
    window.po.ignore(ignoring);
  }
});

$('#tab').addEventListener('click', () => expand());
$('#b-wins').addEventListener('click', () => show('windows'));
$('#b-rec').addEventListener('click', () => show('rec'));
$('#b-model').addEventListener('click', () => show('model'));
$('#b-set').addEventListener('click', () => show('settings'));
$('#b-attached').addEventListener('click', () => show(live().length ? 'rec' : 'windows'));
$('#b-more').addEventListener('click', () => window.po.menu());

body.addEventListener('click', (e) => {
  const t = e.target.closest('[data-close],[data-detach],[data-attach],[data-recstop],[data-open],[data-dev],[data-ds],[data-dsnew],[data-model],[data-set]');
  if (!t) return;
  if (t.dataset.detach) { e.stopPropagation(); detach(t.dataset.detach); }
  else if (t.dataset.close !== undefined) collapse();
  else if (t.dataset.attach) attach(t.dataset.attach);
  else { kb = rows().indexOf(t); activate(t); render(); }
});
body.addEventListener('input', (e) => {
  if (e.target.id === 'q') { query = e.target.value; kb = 0; render(); body.scrollTop = 0; }
});

function openSearch() {
  if (query === null) query = '';
  kb = 0;
  sig = '';
  render();
  const q = $('#q');
  if (q) { q.focus(); q.select(); }
}

function closeSearch() {
  query = null;
  kb = 0;
  sig = '';
  if (document.activeElement) document.activeElement.blur();
  render();
}

// keys while the search box has focus: it keeps the letters, so moving is arrows or Ctrl+j/k/n/p
function searchKey(e) {
  const down = e.key === 'ArrowDown' || (e.ctrlKey && (e.key === 'j' || e.key === 'n'));
  const up = e.key === 'ArrowUp' || (e.ctrlKey && (e.key === 'k' || e.key === 'p'));
  if (down || up) { e.preventDefault(); step(down ? 1 : -1); }
  else if (e.key === 'Enter') { const row = selected(); if (row) attach(row.dataset.attach); }
  else if (e.key === 'Escape') { e.preventDefault(); if (query) { query = ''; kb = 0; render(); } else closeSearch(); }
  else if (e.key === 'Tab') e.preventDefault();
}

// keys while typing a dataset name: Enter uses it, Esc leaves the box
function datasetKey(e) {
  if (e.key === 'Enter') {
    const v = e.target.value.trim();
    if (v) { window.po.setDataset(v); e.target.value = ''; }
    e.target.blur();
  } else if (e.key === 'Escape') { e.preventDefault(); e.target.blur(); }
  else if (e.key === 'Tab') e.preventDefault();
}

document.addEventListener('keydown', (e) => {
  if (!expanded) return;
  if (e.target.id === 'q') return searchKey(e);
  if (e.target.id === 'ds') return datasetKey(e);
  if (e.key === 'Escape') { if (query !== null) closeSearch(); else if (view !== 'windows') show('windows'); else collapse(); return; }
  // h/l (and Tab) walk the side rail: windows, models, record
  const VIEWS = ['windows', 'model', 'rec', 'settings'];
  if (e.key === 'Tab' || e.key === 'l' || e.key === 'h') {
    e.preventDefault();
    const d = e.key === 'h' || e.shiftKey ? -1 : 1;
    show(VIEWS[(VIEWS.indexOf(view) + d + VIEWS.length) % VIEWS.length]);
    return;
  }
  if (e.key === 'ArrowDown' || e.key === 'j') { e.preventDefault(); step(1); }
  else if (e.key === 'ArrowUp' || e.key === 'k') { e.preventDefault(); step(-1); }
  else if (e.key === 'g') { kb = 0; render(); body.scrollTop = 0; }
  else if (e.key === 'G') { kb = rows().length - 1; render(); const r = body.querySelector('.row.kb'); if (r) r.scrollIntoView({ block: 'nearest' }); }
  else if (view === 'rec' || view === 'model' || view === 'settings') {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activate(selected()); }
  }
  else if (e.key === 'Enter') { const row = selected(); if (row) attach(row.dataset.attach); }
  else if (e.key === '/') { e.preventDefault(); openSearch(); }
  else if (e.key === 'x') { const row = selected(); if (row) detach(row.dataset.attach); }
  else if (e.key === 'r') { const row = selected(); if (row) record(row.dataset.attach); }
  else if (e.key === 'b') { const row = selected(); if (row) play(row.dataset.attach); }
});

window.po.on('state', (s) => { S = s; if (wantPrev && expanded && query === null && view === 'windows') selectPrev(); render(); });
window.po.on('collapse', collapse);
window.po.on('expand', (prev) => expand(prev));
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3500);
}
window.po.on('toast', toast);
// recording clocks tick every second while the panel is open
setInterval(() => { if (expanded && live().length) render(); }, 1000);
setInterval(() => { if (expanded) render(); }, 30000);
window.po.ready();
