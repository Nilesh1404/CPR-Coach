"""Audio: one output stream that mixes a gap-free 110 BPM beat with spoken lines,
and a microphone reader with a simple energy VAD or push-to-talk.

The beat is generated inside the sound-card callback from a running sample counter,
so nothing the main thread does (speech-to-text, thinking) can delay it.
"""
import queue
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd

SR = 16000


def load_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    elif sw == 4:
        a = np.frombuffer(raw, np.int32).astype(np.float32) / 2 ** 31
    else:
        a = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    if sr != SR:
        a = np.interp(np.arange(0, len(a), sr / SR), np.arange(len(a)), a).astype(np.float32)
    return a


AUDIO_EXT = {".wav", ".ogg", ".opus", ".mp3", ".m4a", ".aac", ".flac", ".webm", ".amr", ".3gp"}


def load_any(path):
    """WAV directly; anything else (WhatsApp .ogg/.opus, phone .m4a, .mp3) through ffmpeg -> 16 kHz mono."""
    import subprocess
    import tempfile
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            return load_wav(path)
        except Exception:
            pass                                   # odd WAV encodings: let ffmpeg read it
    tmp = Path(tempfile.mkdtemp()) / "a.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), str(tmp)],
                   check=True)
    return load_wav(tmp)


def save_wav(path, audio, sr=SR):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    a = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(a.tobytes())


class Player:
    def __init__(self, bpm=110, beat_vol=0.35, device=None):
        self.period = int(round(SR * 60 / bpm))
        t = np.arange(int(0.03 * SR)) / SR
        self.click = (np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 120)).astype(np.float32)
        self.pos = 0
        self.beat_on = False
        self.beat_vol = beat_vol
        self.duck = 1.0                       # set <1 to lower the beat while listening
        self.queue = deque()
        self.cur, self.cur_i = None, 0
        self.last_audio_end = 0.0
        self.beats_played = 0
        self.stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32",
                                      blocksize=256, latency="low", device=device,
                                      callback=self._cb)

    def _cb(self, out, frames, _t, _status):
        buf = np.zeros(frames, np.float32)
        if self.beat_on:
            idx = (self.pos + np.arange(frames)) % self.period
            m = idx < len(self.click)
            buf[m] += self.click[idx[m]] * self.beat_vol * self.duck
            self.beats_played += int(np.sum(idx == 0))
        self.pos += frames
        filled = 0
        while filled < frames:
            if self.cur is None:
                try:
                    self.cur, self.cur_i = self.queue.popleft(), 0
                except IndexError:
                    break
            n = min(frames - filled, len(self.cur) - self.cur_i)
            buf[filled:filled + n] += self.cur[self.cur_i:self.cur_i + n]
            filled += n
            self.cur_i += n
            if self.cur_i >= len(self.cur):
                self.cur = None
        if filled:
            self.last_audio_end = time.time()
        out[:, 0] = np.clip(buf, -1, 1)

    def start(self):
        self.stream.start()

    def stop(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

    def play(self, audio):
        self.queue.append(audio.astype(np.float32))
        self.queue.append(np.zeros(int(0.25 * SR), np.float32))

    def speaking(self, tail=0.3):
        return self.cur is not None or bool(self.queue) or time.time() - self.last_audio_end < tail

    def wait_done(self):
        while self.speaking(tail=0):
            time.sleep(0.02)


class Mic:
    FRAME = 480   # 30 ms

    def __init__(self, device=None):
        self.q = queue.Queue()
        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                     blocksize=self.FRAME, device=device, callback=self._cb)

    def _cb(self, data, frames, _t, _status):
        self.q.put(data[:, 0].copy())

    def start(self):
        self.stream.start()

    def drain(self):
        while not self.q.empty():
            self.q.get_nowait()

    def stop(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

    def listen_vad(self, player=None, half_duplex=True, max_s=8.0, silence_s=0.7,
                   min_rms=0.008, ratio=3.0, duck=None, stop=None, level_cb=None):
        """Block until one utterance is heard; return float32 audio.
        stop: threading.Event -> returns None when set. level_cb(rms): called every frame."""
        floor = 0.003
        pre = deque(maxlen=10)        # 300 ms pre-roll
        recent = deque(maxlen=10)
        speech, in_speech, silent = [], False, 0
        while True:
            try:
                f = self.q.get(timeout=0.1)
            except queue.Empty:
                if stop is not None and stop.is_set():
                    return None
                continue
            if stop is not None and stop.is_set():
                return None
            rms = float(np.sqrt(np.mean(f * f)))
            if level_cb:
                level_cb(rms)
            if half_duplex and player is not None and player.speaking():
                pre.clear(); recent.clear()
                continue                         # ignore our own voice
            thr = max(floor * ratio, min_rms)
            loud = rms > thr
            if not in_speech:
                floor = 0.97 * floor + 0.03 * rms if not loud else floor
                pre.append(f)
                recent.append(loud)
                if sum(recent) >= 7:             # ~210 ms of the last 300 ms loud -> speech, not a click
                    in_speech, speech, silent = True, list(pre), 0
                    if duck is not None and player is not None:
                        player.duck = duck
            else:
                speech.append(f)
                silent = 0 if loud else silent + 1
                if silent * 0.03 >= silence_s or len(speech) * 0.03 >= max_s:
                    if player is not None:
                        player.duck = 1.0
                    return np.concatenate(speech)

    def listen_ptt(self, prompt="  [Enter] = talk, [Enter] = done > "):
        input(prompt)
        self.drain()
        stop = threading.Event()
        chunks = []

        def grab():
            while not stop.is_set():
                try:
                    chunks.append(self.q.get(timeout=0.1))
                except queue.Empty:
                    pass
        th = threading.Thread(target=grab, daemon=True)
        th.start()
        input("  recording... [Enter] to stop ")
        stop.set()
        th.join()
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


class Voices:
    """Pre-made WAVs: <dir>/<lang>/<Lxx>.wav, falling back to English, then to a chime."""

    def __init__(self, roots):
        """roots: one folder or comma-separated folders, tried in order."""
        self.roots = [Path(r) for r in str(roots).split(",")]
        self.cache = {}
        t = np.arange(int(0.15 * SR)) / SR
        self.chime = np.concatenate([0.3 * np.sin(2 * np.pi * 660 * t), 0.3 * np.sin(2 * np.pi * 880 * t)])

    def get(self, lid, lang):
        for lg in (lang, "en"):
            for root in self.roots:
                p = root / lg / f"{lid}.wav"
                if p.exists():
                    if p not in self.cache:
                        self.cache[p] = load_wav(p)
                    return self.cache[p], lg
        return self.chime.astype(np.float32), None
