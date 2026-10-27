// Node side of native/rec (player-one-rec.exe): one process kept running for the app's life,
// because its RawInput tap and gamepad poll are process-wide. Requests and replies are JSON lines
// matched by id; unprompted lines are events ("log", "finished").

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');

const NATIVE = path.join(__dirname, '..', 'native', 'target');

function exePath() {
  for (const p of ['release', 'debug']) {
    const f = path.join(NATIVE, p, 'player-one-rec.exe').replace('app.asar', 'app.asar.unpacked');
    if (fs.existsSync(f)) return f;
  }
  return null;
}

// The sidecar's stderr and its event lines, for debugging play; restarted past 5 MB.
function openLog(file) {
  if (!file) return () => {};
  try { if (fs.statSync(file).size > 5e6) fs.rmSync(file); } catch {}
  return (kind, text) => {
    const t = new Date().toISOString().slice(11, 23);
    const lines = text.split('\n').filter((l) => l.trim());
    try { fs.appendFileSync(file, lines.map((l) => `${t} ${kind} ${l.trimEnd()}\n`).join('')); } catch {}
  };
}

class Recorder extends EventEmitter {
  constructor(logFile) {
    super();
    this.log = openLog(logFile);
    this.proc = null;
    this.buf = '';
    this.waits = new Map();
    this.n = 0;
  }

  start() {
    if (this.proc) return true;
    const exe = exePath();
    if (!exe) return false;
    const proc = spawn(exe, [], { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
    this.proc = proc;
    proc.stdout.setEncoding('utf8');
    proc.stdout.on('data', (d) => {
      this.buf += d;
      let i;
      while ((i = this.buf.indexOf('\n')) >= 0) {
        const line = this.buf.slice(0, i).trim();
        this.buf = this.buf.slice(i + 1);
        if (!line) continue;
        let m;
        try { m = JSON.parse(line); } catch { continue; }
        if (m.event) { this.log('event', line); this.emit(m.event, m); continue; }
        const w = this.waits.get(m.id);
        if (w) { this.waits.delete(m.id); w(m); }
      }
    });
    proc.stderr.setEncoding('utf8');
    proc.stderr.on('data', (d) => this.log('stderr', d));
    setImmediate(() => this.emit('spawn'));
    proc.on('exit', (code) => {
      this.log('exit', `code ${code}`);
      if (this.proc === proc) this.proc = null;
      for (const w of this.waits.values()) w({ ok: false, error: 'recorder exited' });
      this.waits.clear();
      this.emit('exit');
    });
    return true;
  }

  request(obj, timeoutMs = 10000) {
    if (!this.start()) return Promise.resolve({ ok: false, error: 'recorder not built: run `cargo build --release` in native/' });
    const id = ++this.n;
    return new Promise((resolve) => {
      const timer = setTimeout(() => { this.waits.delete(id); resolve({ ok: false, error: 'timeout' }); }, timeoutMs);
      this.waits.set(id, (v) => { clearTimeout(timer); resolve(v); });
      this.proc.stdin.write(`${JSON.stringify({ ...obj, id })}\n`);
    });
  }

  devices() { return this.request({ op: 'devices' }); }

  setDevices(mask) { return this.request({ op: 'set_devices', mask }); }

  passthrough(on) { return this.request({ op: 'passthrough', on: !!on }); }

  padMenu(on) { return this.request({ op: 'pad_menu', on: !!on }); }

  begin(opts) { return this.request({ op: 'start', ...opts }); }

  end(hwnd) { return this.request({ op: 'stop', hwnd }); }

  // closing stdin makes it stop and finalize every recording before it exits
  stop() { if (this.proc) this.proc.stdin.end(); }
}

module.exports = { Recorder };
