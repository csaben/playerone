// Node side of win-helper.ps1: one PowerShell kept warm so listing and typing are instant.

const { spawn } = require('child_process');
const path = require('path');

class Helper {
  constructor() {
    this.proc = null;
    this.buf = '';
    this.queue = [];
  }

  start() {
    if (this.proc) return;
    const script = path.join(__dirname, 'win-helper.ps1').replace('app.asar', 'app.asar.unpacked');
    const proc = spawn('powershell.exe', ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', script],
      { windowsHide: true });
    this.proc = proc;
    proc.stdout.setEncoding('utf8');
    proc.stdout.on('data', (d) => {
      this.buf += d;
      let i;
      while ((i = this.buf.indexOf('\n')) >= 0) {
        const line = this.buf.slice(0, i).trim();
        this.buf = this.buf.slice(i + 1);
        if (!line) continue;
        const w = this.queue.shift();
        if (!w) continue;
        try { w.resolve(JSON.parse(line)); } catch { w.resolve({ ok: false, error: line }); }
      }
    });
    proc.on('exit', () => {
      if (this.proc === proc) this.proc = null;
      for (const w of this.queue.splice(0)) w.resolve({ ok: false, error: 'helper exited' });
    });
  }

  request(obj, timeoutMs = 5000) {
    this.start();
    return new Promise((resolve) => {
      const w = { resolve };
      const timer = setTimeout(() => {
        const i = this.queue.indexOf(w);
        if (i >= 0) this.queue.splice(i, 1);
        resolve({ ok: false, error: 'timeout' });
      }, timeoutMs);
      w.resolve = (v) => { clearTimeout(timer); resolve(v); };
      this.queue.push(w);
      this.proc.stdin.write(`${JSON.stringify(obj)}\n`);
    });
  }

  list(self) { return this.request({ op: 'list', self }); }

  fg() { return this.request({ op: 'fg' }, 1000); }

  type(hwnd, text, delay = 12) { return this.request({ op: 'type', hwnd, text, delay }, 5000 + text.length * (delay + 20)); }

  stop() { if (this.proc) this.proc.kill(); }
}

module.exports = { Helper };
