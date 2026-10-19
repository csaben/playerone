// Ships recorded chunks to the trainer's ingest server as they close, one at a time, in order.
// Contract (train/ingest.py): PUT <ingest>/chunks/<session>/<file>, .meta.json last = complete.
// Failed sends retry with backoff; loose files are deleted once the trainer has them (the
// session's shard keeps its own copy).
const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');

const PARTS = ['.p0.mp4', '.p0.jsonl', '.events.jsonl', '.meta.json'];

class Shipper extends EventEmitter {
  constructor() {
    super();
    this.queue = [];
    this.busy = false;
  }

  add(url, chunk) {
    this.queue.push({ url: url.replace(/\/+$/, ''), ...chunk, tries: 0 });
    this.pump();
  }

  get pending() { return this.queue.length; }

  async pump() {
    if (this.busy) return;
    this.busy = true;
    while (this.queue.length) {
      const c = this.queue[0];
      try {
        await this.send(c);
        this.queue.shift();
        this.emit('sent', c);
      } catch (e) {
        c.tries++;
        this.emit('failed', c, e);
        await new Promise((r) => setTimeout(r, Math.min(30000, 1000 * 2 ** Math.min(c.tries, 5))));
      }
    }
    this.busy = false;
  }

  async send({ url, session, key, dir }) {
    for (const ext of PARTS) {
      const file = `${key}${ext}`;
      const body = fs.readFileSync(path.join(dir, file));
      const res = await fetch(`${url}/chunks/${encodeURIComponent(session)}/${encodeURIComponent(file)}`, {
        method: 'PUT', body, signal: AbortSignal.timeout(30000),
      });
      if (!res.ok) throw new Error(`${res.status} ${await res.text().catch(() => '')}`.trim());
    }
    for (const ext of PARTS) fs.rmSync(path.join(dir, `${key}${ext}`), { force: true });
  }
}

module.exports = { Shipper };
