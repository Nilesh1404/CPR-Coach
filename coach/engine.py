"""The live coach as a background service, used by the web front end (server.py).

Owns the speaker (beat + lines), the microphone, speech-to-text and the dialog, and
publishes everything that happens as events the browser can follow.
"""
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .dialog import Dialog, load_script
from .nlu import NLU

ROOT = Path(__file__).resolve().parent.parent
VOICE_ROOTS = [ROOT / "voices" / "human", ROOT / "voices" / "female", ROOT / "voices" / "synth"]


class Engine:
    def __init__(self):
        self.script = load_script()
        self.nlu = NLU()
        self.dialog = Dialog(nlu=self.nlu, script=self.script)
        self.cond = threading.Condition()
        self.events, self.history, self.seq = [], [], 0
        self.status = "idle"            # idle, loading, listening, thinking, speaking, error
        self.running = False
        self.mode = "vad"               # vad | ptt
        self.stt_spec, self.stt_lang, self.stt = None, None, None
        self.player = self.mic = self.voices = None
        self.ptt_down = False
        self.stop_ev = threading.Event()
        self.thread = None
        self.lock = threading.Lock()
        self._last_level = 0.0

    # ---------------- events ----------------
    def emit(self, kind, **data):
        with self.cond:
            self.seq += 1
            ev = {"id": self.seq, "type": kind, "t": round(time.time(), 3), **data}
            self.events.append(ev)
            if len(self.events) > 400:
                self.events = self.events[-400:]
            if kind in ("heard", "reply", "system"):
                self.history.append(ev)
                self.history = self.history[-200:]
            self.cond.notify_all()
        return ev

    def wait_events(self, after, timeout=15.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq > after, timeout)
            return [e for e in self.events if e["id"] > after]

    def set_status(self, s):
        if s != self.status:
            self.status = s
            self.emit("status", status=s)

    def snapshot(self):
        return {"status": self.status, "running": self.running, "mode": self.mode,
                "state": self.dialog.state, "lang": self.dialog.lang, "beat": self.beat_on(),
                "stt": self.stt_spec, "stt_lang": self.stt_lang, "history": self.history,
                "seq": self.seq, "started": bool(self.dialog.last_reply)}

    def beat_on(self):
        return self.dialog.state in ("compressions", "confirming") and bool(self.dialog.last_reply)

    # ---------------- control ----------------
    def start(self, stt="whisper:small@10", mode="vad", lang="auto3"):
        if self.running:
            return
        self.mode = mode
        # Language chosen in the UI (not Auto): the whole call is in that language - replies too,
        # not only speech-to-text. Whisper writes Marathi in Hindi spelling, so text-based
        # detection alone often keeps answering in Hindi.
        with self.lock:
            self.dialog.fixed_lang = lang in ("hi", "mr", "en")
            if self.dialog.fixed_lang:
                self.dialog.lang = lang
        self.stop_ev.clear()
        self.running = True
        self.thread = threading.Thread(target=self._run, args=(stt, lang), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_ev.set()
        if self.thread:
            self.thread.join(timeout=3)
        for dev in (self.player, self.mic):
            if dev:
                dev.stop()
        self.player = self.mic = None
        self.running = False
        self.set_status("idle")
        self.emit("system", text="Stopped. Microphone and speaker are off.")

    def reset(self):
        with self.lock:
            fixed = getattr(self.dialog, "fixed_lang", False)
            lang = self.dialog.lang
            self.dialog = Dialog(nlu=self.nlu, script=self.script, lang=lang)
            self.dialog.fixed_lang = fixed
            if hasattr(self.stt, "reset"):
                self.stt.reset()
        with self.cond:
            self.history = []
        self.emit("system", text="New session.")
        if self.player:
            self.player.beat_on = False
        if self.running:
            self._reply(self.dialog.start(), None, source="start")

    def set_mode(self, mode):
        self.mode = mode
        self.emit("mode", mode=mode)

    def set_ptt(self, down):
        self.ptt_down = bool(down)

    # ---------------- main loop ----------------
    def _run(self, stt, lang):
        try:
            self.set_status("loading")
            from .audio import Mic, Player, Voices
            if self.stt is None or self.stt_spec != stt or self.stt_lang != lang:
                self.emit("system", text=f"Loading {stt} …")
                from .stt import make_stt
                t0 = time.perf_counter()
                self.stt = make_stt(stt, lang=lang)
                self.stt_spec, self.stt_lang = stt, lang
                self.emit("system", text=f"{stt} ready in {time.perf_counter() - t0:.1f} s.")
            self.voices = Voices(",".join(str(r) for r in VOICE_ROOTS))
            self.player, self.mic = Player(), Mic()
            self.player.start()
            self.mic.start()
        except Exception as e:
            self.running = False
            self.set_status("error")
            self.emit("system", text=f"Could not start: {e}", error=True)
            return

        if not self.dialog.last_reply:
            self._reply(self.dialog.start(), None, source="start")
        else:
            self.player.beat_on = self.beat_on()
        self.set_status("listening")
        while not self.stop_ev.is_set():
            try:
                audio = (self.mic.listen_vad(self.player, stop=self.stop_ev, level_cb=self._level)
                         if self.mode == "vad" else self._ptt_capture())
                if audio is None or len(audio) < 1600:
                    continue
                self._process_audio(audio)
            except Exception as e:   # keep running, keep the beat
                self.emit("system", text=f"Error: {e}", error=True)
                self.set_status("listening")

    def _level(self, rms):
        now = time.time()
        if now - self._last_level > 0.1:
            self._last_level = now
            self.emit("level", rms=round(rms, 4))
            if self.status in ("listening", "speaking") and self.player:
                self.set_status("speaking" if self.player.speaking(0) else "listening")

    def _ptt_capture(self):
        chunks = []
        while not self.stop_ev.is_set():
            try:
                f = self.mic.q.get(timeout=0.1)
            except queue.Empty:
                continue
            self._level(float(np.sqrt(np.mean(f * f))))
            if self.ptt_down:
                chunks.append(f)
            elif chunks:
                return np.concatenate(chunks)
        return None

    def _process_audio(self, audio):
        self.set_status("thinking")
        s = self.stt.transcribe(audio, hint=self.dialog.lang)   # spelling hint in the current language
        self.handle_text(s["text"], stt=s, audio_s=round(len(audio) / 16000, 2))

    # ---------------- shared by voice and typed input ----------------
    def handle_text(self, text, stt=None, audio_s=None):
        with self.lock:
            if not self.dialog.last_reply:
                self._reply(self.dialog.start(), None, source="start")
            if stt and stt.get("lang") in ("hi", "mr", "en") and not getattr(self.dialog, "fixed_lang", False):
                self.dialog.lang = stt["lang"]          # the winning model's language is the best guess
            r = self.dialog.handle(text, stt_logprob=stt["logprob"] if stt else None)
        a = r["analysis"]
        self.emit("heard", text=text, source="voice" if stt else "typed",
                  stt_ms=stt["ms"] if stt else None, logprob=stt["logprob"] if stt else None,
                  stt_lang=stt.get("lang") if stt else None, audio_s=audio_s)
        self._reply(r["reply"], r, source="dialog")
        return r

    def _reply(self, ids, r, source):
        lang = self.dialog.lang
        lines = [{"id": i, "text": self.dialog.line_text(i), "en": self.script["lines"][i]["en"]}
                 for i in ids]
        a = (r or {}).get("analysis", {})
        self.emit("reply", lines=lines, state=self.dialog.state, lang=lang, beat=self.beat_on(),
                  intent=(r or {}).get("intent"), escalate=(r or {}).get("escalate", False),
                  nlu_ms=(r or {}).get("ms"), score=a.get("score"), nearest=a.get("nearest"),
                  knn_intent=a.get("knn_intent"), flags=a.get("flags"),
                  confident_stop=a.get("confident_stop"), source=source)
        if self.player:
            self.player.beat_on = self.beat_on()
            missing = []
            for i in ids:
                audio, used = self.voices.get(i, lang)
                if used != lang:
                    missing.append(i)
                self.player.play(audio)
            if missing:
                self.emit("system", text=f"No {lang} recording for {', '.join(missing)}; "
                                         f"played {'English' if used else 'a chime'} instead.")
            self.set_status("speaking")
