"""Local web front end for the CPR coach. Runs fully offline.

  python server.py            # opens http://127.0.0.1:8765 in your browser
  python server.py --no-open --port 9000
"""
import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from coach.dialog import Dialog, START_CPR  # noqa: E402
from coach.engine import Engine, VOICE_ROOTS  # noqa: E402

UI = ROOT / "ui"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png"}
engine = Engine()


def line_info(ids, lang):
    s = engine.script["lines"]
    return [{"id": i, "text": s[i].get(lang) or s[i]["en"], "en": s[i]["en"]} for i in ids]


def text_test(text, state):
    t0 = time.perf_counter()
    try:
        d = Dialog(nlu=engine.nlu, script=engine.script)
        r = d.handle(text, state=state)
        a = r["analysis"]
        return {"intent": r["intent"], "reply": r["reply"], "escalate": r["escalate"],
                "state_after": r["state"], "lang": r["lang"], "lines": line_info(r["reply"], r["lang"]),
                "score": a.get("score"), "nearest": a.get("nearest"), "knn_intent": a.get("knn_intent"),
                "flags": a.get("flags", {}), "confident_stop": a.get("confident_stop"),
                "ms": round((time.perf_counter() - t0) * 1000, 2)}
    except Exception as e:
        reply = START_CPR if state == "checking" else ["L15"]
        return {"intent": "unclear", "reply": reply, "escalate": False, "error": repr(e),
                "lines": line_info(reply, "hi"), "ms": round((time.perf_counter() - t0) * 1000, 2)}


def config():
    langs = ["en", "hi", "mr"]
    voices = {}
    for lid in engine.script["lines"]:
        voices[lid] = {}
        for lg in langs:
            kind = None
            for root in VOICE_ROOTS:
                if (root / lg / f"{lid}.wav").exists():
                    kind = root.name
                    break
            voices[lid][lg] = kind
    return {"lines": engine.script["lines"], "voices": voices,
            "script_version": engine.script.get("version"),
            # best first; measured on 93 real clips (reports/), small@10 = small with a 10 s window
            "models": [["indic@10", "Hindi+Marathi models - best: 82/93 right"],
                       ["whisper:small@10", "Whisper small@10 - 54/93 right, 0 unsafe, max 7 s"],
                       ["whisper:small", "small - same accuracy, can freeze up to 17 s"],
                       ["whisper:base", "base - faster, 29/93 right"],
                       ["whisper:tiny", "tiny - fastest, 14/93 right (weak devices only)"]],
            "stt_langs": ["auto3", "hi", "mr", "en"]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/state":
            return self._json(engine.snapshot())
        if u.path == "/api/config":
            return self._json(config())
        if u.path == "/api/events":
            return self._sse(int(parse_qs(u.query).get("after", ["0"])[0]))
        path = UI / ("index.html" if u.path in ("/", "") else u.path.lstrip("/").replace("ui/", "", 1))
        path = path.resolve()
        if UI.resolve() not in path.parents or not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _sse(self, after):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = int(self.headers.get("Last-Event-ID") or after)
        try:
            while True:
                evs = engine.wait_events(last, timeout=10)
                if not evs:
                    self.wfile.write(b": ping\n\n")
                for ev in evs:
                    self.wfile.write(f"id: {ev['id']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
                                     .encode("utf-8"))
                    last = ev["id"]
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def do_POST(self):
        u = urlparse(self.path).path
        b = self._body()
        if u == "/api/start":
            known = [m[0] for m in config()["models"]]
            stt = str(b.get("stt", "")).split(",")[0].strip()      # old cached pages sent "id,label"
            if stt not in known:
                stt = known[0]
            lang = b.get("lang", "auto3") if b.get("lang") in ("auto3", "hi", "mr", "en") else "auto3"
            engine.start(stt=stt, mode=b.get("mode", "vad"), lang=lang)
            return self._json({"ok": True})
        if u == "/api/stop":
            threading.Thread(target=engine.stop, daemon=True).start()
            return self._json({"ok": True})
        if u == "/api/reset":
            engine.reset()
            return self._json({"ok": True})
        if u == "/api/mode":
            engine.set_mode(b.get("mode", "vad"))
            return self._json({"ok": True})
        if u == "/api/ptt":
            engine.set_ptt(b.get("down"))
            return self._json({"ok": True})
        if u == "/api/say":
            r = engine.handle_text(b.get("text", ""))
            return self._json({"intent": r["intent"], "reply": r["reply"]})
        if u == "/api/test":
            return self._json(text_test(b.get("text", ""), b.get("state", "compressions")))
        if u == "/api/batch":
            out = []
            for it in b.get("items", []):
                r = text_test(it.get("text", ""), it.get("state", "compressions"))
                out.append({"id": it.get("id"), "intent": r["intent"], "reply": r["reply"],
                            "escalate": r["escalate"], "ms": r["ms"]})
            return self._json({"results": out})
        self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}"
    print(f"CPR coach front end: {url}   (Ctrl+C to quit)")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if engine.running:
            engine.stop()


if __name__ == "__main__":
    main()
