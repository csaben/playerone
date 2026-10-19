"""Chunk ingest server. Stdlib only.

PUT /chunks/<session_id>/<filename>  raw body -> $DATA/sessions/<session_id>/<filename> (atomic)
GET /healthz                         -> {"ok":true,"sessions":N,"chunks":M}

A chunk is complete once its .meta.json exists (the client uploads it last).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SUFFIXES = (".p0.mp4", ".p0.jsonl", ".events.jsonl", ".meta.json")
MAX_BODY = 512 * 1024 * 1024
DEFAULT_PORT = 8961  # 8960 is taken on cradle by ml-dash


def data_dir() -> Path:
    return Path(os.environ.get("DATA", "~/player-one-data")).expanduser()


def valid_name(s: str) -> bool:
    return bool(NAME_RE.match(s)) and s not in (".", "..")


LAST_PUT: dict = {}
STALE_S = 60  # a component whose status file is older than this is reported as "down"


def _read_status(root: Path, name: str) -> dict:
    p = root / "status" / f"{name}.json"
    try:
        st = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {"state": "down", "detail": f"no {name} status file"}
    age = time.time() - st.get("updated_at", 0)
    st["age_s"] = round(age, 1)
    if age > STALE_S:
        st["state"] = "down"
        st["detail"] = f"status not updated for {age:.0f}s"
    return st


def status(root: Path) -> dict:
    """One-stop view of the cradle loop for the dock: ingest + trainer + policy server."""
    s, c = counts(root)
    metas = list((root / "sessions").glob("*/*.meta.json")) if (root / "sessions").is_dir() else []
    newest = max(metas, key=lambda p: p.stat().st_mtime) if metas else None
    frames = 0
    for m in metas:
        try:
            frames += int(json.loads(m.read_text()).get("frame_count") or 0)
        except (OSError, ValueError):
            pass
    ingest = {"state": "up", "sessions": s, "chunks": c, "frames": frames,
              "minutes": round(frames / 20 / 60, 1),
              "last_chunk_at": newest.stat().st_mtime if newest else None,
              "last_chunk": f"{newest.parent.name}/{newest.name[:-len('.meta.json')]}" if newest else None,
              "last_put": LAST_PUT or None}
    models = {p.name[len("serve-"):-len(".json")]: _read_status(root, p.name[:-len(".json")])
              for p in sorted((root / "status").glob("serve-*.json"))} if (root / "status").is_dir() else {}
    return {"ok": True, "now": time.time(), "ingest": ingest,
            "train": _read_status(root, "train"), "vpttrain": _read_status(root, "vpttrain"),
            "serve": _read_status(root, "serve"),
            "models": models}  # every policy server by name ("bc", "vpt", ...), each with policy/port/version


def counts(root: Path) -> tuple[int, int]:
    sess = root / "sessions"
    if not sess.is_dir():
        return 0, 0
    n_s = n_c = 0
    for d in sess.iterdir():
        if d.is_dir():
            n_s += 1
            n_c += sum(1 for _ in d.glob("*.meta.json"))
    return n_s, n_c


def make_handler(root: Path):
    class H(BaseHTTPRequestHandler):
        server_version = "p1ingest/0.1"

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            route = self.path.split("?")[0]
            if route == "/healthz":
                s, c = counts(root)
                return self._json(200, {"ok": True, "sessions": s, "chunks": c})
            if route == "/status":
                return self._json(200, status(root))
            self._json(404, {"ok": False, "error": "not found"})

        def do_PUT(self):  # noqa: N802
            parts = self.path.split("?")[0].strip("/").split("/")
            if len(parts) != 3 or parts[0] != "chunks":
                return self._json(404, {"ok": False, "error": "expected /chunks/<session_id>/<filename>"})
            sid, fname = parts[1], parts[2]
            if not (valid_name(sid) and valid_name(fname)):
                return self._json(400, {"ok": False, "error": "bad session_id or filename"})
            if not fname.endswith(SUFFIXES):
                return self._json(400, {"ok": False, "error": f"filename must end with one of {SUFFIXES}"})
            length = self.headers.get("Content-Length")
            if length is None:
                return self._json(411, {"ok": False, "error": "Content-Length required"})
            n = int(length)
            if n < 0 or n > MAX_BODY:
                return self._json(413, {"ok": False, "error": "body too large"})
            dest_dir = root / "sessions" / sid
            dest_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=dest_dir)
            try:
                with os.fdopen(fd, "wb") as f:
                    left = n
                    while left > 0:
                        buf = self.rfile.read(min(left, 1 << 20))
                        if not buf:
                            raise ConnectionError("short body")
                        f.write(buf)
                        left -= len(buf)
                os.replace(tmp, dest_dir / fname)
                LAST_PUT.update(at=time.time(), session=sid, file=fname, bytes=n)
            except Exception as e:  # noqa: BLE001
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                return self._json(400, {"ok": False, "error": str(e)})
            self._json(200, {"ok": True})

        def log_message(self, fmt, *args):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    return H


def serve(host: str, port: int, root: Path) -> ThreadingHTTPServer:
    root.mkdir(parents=True, exist_ok=True)
    return ThreadingHTTPServer((host, port), make_handler(root))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("P1_INGEST_PORT", DEFAULT_PORT)))
    a = ap.parse_args()
    root = data_dir()
    srv = serve(a.host, a.port, root)
    print(f"ingest listening on {a.host}:{a.port}, DATA={root}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
