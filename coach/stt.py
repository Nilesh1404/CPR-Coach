"""Speech-to-text wrappers. Offline once the model files are downloaded.

whisper:<size>  -> faster-whisper (CTranslate2, int8 on CPU). Sizes: tiny, base, small, medium...
Add other engines (Vosk, IndicConformer) here with the same transcribe() signature.

lang: "hi"/"mr"/"en" forces one language; "auto" lets Whisper pick from all 99;
"auto3" (default) picks the best of hi/mr/en only, so short noisy clips are not
mistaken for Urdu, Nepali, etc.
"""
import os
import time

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")    # never touch the network during a session
# Intel MKL reserves ~2.7 GB of Windows commit memory PER model (only ~0.35 GB is used). Two models
# then reserve ~5 GB and loading fails on an 8 GB laptop ("mkl_malloc: failed to allocate memory").
# Without MKL: ~1.1 GB for both models, about 20% slower decoding. Must be set before ctranslate2 loads.
os.environ.setdefault("CT2_USE_MKL", "0")

# Biases the decoder towards the words that change the instruction.
PROMPT = "साँस नहीं ले रहा। श्वास घेत नाही। थांबा। हिल रहा। breathing, not breathing."
# Per-language hints, chosen from the conversation's current language. They teach Whisper how the
# words people really say are spelled (on 60 Marathi clips: intent 30 -> 43 right).
# SAFETY: a hint must never contain the stop phrase ("श्वास घेतोय", "साँस ले रहा है", "breathing
# normally") - with it in the hint, Whisper turned a real "श्वास घेत नाही" into "श्वास घेतोय".
PROMPTS = {
    "mr": "श्वास घेत नाही. नाही, अजून नाही. धापा टाकतोय, विचित्र आवाज येतोय. काहीच प्रतिसाद नाही, "
          "हलत नाही. तो हलतोय. हात दुखतायत, मी थकलो. हात कुठे ठेवू? किती जोरात दाबू? "
          "ॲम्ब्युलन्सला फोन केला. थांबा.",
    "hi": "साँस नहीं ले रहा। नहीं, अभी भी नहीं। अजीब साँस, हाँफ रहा है। हिल ही नहीं रहा, जवाब नहीं दे रहा। "
          "आँख खोल रहा है। हाथ थक गए। हाथ कहाँ रखूँ? कितना ज़ोर से दबाऊँ? एम्बुलेंस को फोन किया। रुको।",
    "en": "He is not breathing. Still not breathing. Gasping, strange breathing. Not responding, not moving. "
          "He moved. My arms are tired. Where do I put my hands? How hard do I push? Called the ambulance.",
}
ALLOWED = ("hi", "mr", "en")


class WhisperSTT:
    def __init__(self, size="small", lang="auto3", threads=4, prompt=PROMPT, chunk_s=None):
        from faster_whisper import WhisperModel
        self.name = f"whisper-{size}"
        self.lang = lang
        self.prompt = prompt
        self.chunk_s = chunk_s
        self.model = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=threads)

    sticky = None   # auto3: language found on an earlier utterance, reused to save a pass

    def _pick_lang(self, audio, force_detect=False):
        if self.lang == "auto":
            return None, None
        if self.lang != "auto3":
            return self.lang, None
        if self.sticky and not force_detect:
            return self.sticky, None
        _, _, probs = self.model.detect_language(audio)
        p = {l: s for l, s in probs if l in ALLOWED}
        best = max(p, key=p.get) if p else "hi"
        self.sticky = best
        return best, round(p.get(best, 0), 3)

    def transcribe(self, audio, sticky=True, hint=None):
        """sticky=False: detect language for this clip alone (used for benchmarks).
        hint: the conversation's current language (hi/mr/en) -> use that language's word hint."""
        self._hint = hint
        t0 = time.perf_counter()
        audio = audio.astype(np.float32)
        if len(audio) < 1600:
            return {"text": "", "logprob": -9.0, "lang": None, "ms": 0.0}
        out = self._run(audio, force_detect=not sticky)
        # Poor confidence with a reused language -> maybe they switched language: retry once.
        if sticky and out["lang_p"] is None and out["logprob"] < -0.9 and self.lang == "auto3":
            retry = self._run(audio, force_detect=True)
            if retry["logprob"] > out["logprob"]:
                out = retry
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    def _run(self, audio, force_detect):
        t0 = time.perf_counter()
        lang, lp_lang = self._pick_lang(audio, force_detect)
        t1 = time.perf_counter()
        kw = dict(language=lang, beam_size=1, temperature=0.0, max_new_tokens=60,
                  initial_prompt=PROMPTS.get(getattr(self, "_hint", None), self.prompt),
                  condition_on_previous_text=False,
                  without_timestamps=True, vad_filter=False)
        if self.chunk_s:
            kw["chunk_length"] = self.chunk_s
        segs, info = self.model.transcribe(audio, **kw)
        segs = [s for s in segs if s.no_speech_prob < 0.6 and s.compression_ratio < 2.4]
        text = " ".join(s.text.strip() for s in segs).strip()
        lp = float(np.mean([s.avg_logprob for s in segs])) if segs else -9.0
        t2 = time.perf_counter()
        return {"text": text, "logprob": round(lp, 3), "lang": info.language, "lang_p": lp_lang,
                "ms": round((t2 - t0) * 1000, 1), "langid_ms": round((t1 - t0) * 1000, 1)}


class IndicSTT:
    """Whisper-small fine-tuned per language (models/whisper-{hindi,marathi}-small-ct2), plus the
    safe word hint. On the 93 real clips: 81/93 intents right vs 54/93 for plain Whisper small.
    Whisper cannot tell Marathi from Hindi by sound, so: decode with the model of the
    conversation's current language; if the words come out in the other language, decode again
    with that language's model (costs a second pass only when the language changes).
    English goes to plain Whisper small, loaded only if needed."""
    PATHS = {"hi": "models/whisper-hindi-small-ct2", "mr": "models/whisper-marathi-small-ct2"}

    def __init__(self, threads=4, chunk_s=10, lang="auto3"):
        from pathlib import Path
        self.root = Path(__file__).resolve().parent.parent
        self.name, self.threads, self.chunk_s, self.lang = "indic", threads, chunk_s, lang
        self.models = {}
        for lg in ([lang] if lang in ("hi", "mr", "en") else ["hi", "mr"]):   # only what is needed
            self._get(lg)

    def _get(self, lang):
        if lang not in self.models:
            from faster_whisper import WhisperModel
            src = str(self.root / self.PATHS[lang]) if lang in self.PATHS else "small"
            self.models[lang] = WhisperModel(src, device="cpu", compute_type="int8", cpu_threads=self.threads)
        return self.models[lang]

    def _run(self, audio, lang):
        segs, _ = self._get(lang).transcribe(
            audio, language=lang, beam_size=1, temperature=0.0, max_new_tokens=60,
            initial_prompt=PROMPTS.get(lang), condition_on_previous_text=False,
            without_timestamps=True, vad_filter=False, chunk_length=self.chunk_s)
        segs = [s for s in segs if s.no_speech_prob < 0.6 and s.compression_ratio < 2.4]
        text = " ".join(s.text.strip() for s in segs).strip()
        lp = float(np.mean([s.avg_logprob for s in segs])) if segs else -9.0
        return text, lp

    UNSURE = -0.5   # avg log-prob below this -> also try the other language's model
    MR_MARGIN = 0.15  # the Marathi model is over-confident even on Hindi speech (median +0.09 on
                      # Hindi, +0.21 on Marathi), so it must beat Hindi by this much. Tuned on the
                      # same 93 clips: Hindi routed 29/33, Marathi 46/60, intent 82/93.

    def transcribe(self, audio, sticky=True, hint=None):
        """Language fixed in the UI -> only that language's model (81/93 on the 93 clips).
        Auto -> both models on the first sentence and whenever the current model is unsure, keep
        the more confident one, then stay with it (79/93, ~4 s/sentence; simulated on the clips).
        The words cannot decide: each model writes any speech in its own language's spelling.
        Never both in parallel: that ran out of memory, and the target laptop has 8 GB."""
        t0 = time.perf_counter()
        audio = audio.astype(np.float32)
        if len(audio) < 1600:
            return {"text": "", "logprob": -9.0, "lang": None, "ms": 0.0}
        if self.lang in ("hi", "mr", "en"):
            text, lp = self._run(audio, self.lang)
            lang, tried = self.lang, 1
        else:
            cur = getattr(self, "chosen", None)
            out = {}
            if cur:
                out[cur] = self._run(audio, cur)
            if not cur or out[cur][1] < self.UNSURE:
                for lg in ("hi", "mr"):
                    if lg not in out:
                        out[lg] = self._run(audio, lg)
            if len(out) == 2:
                lang = "mr" if out["mr"][1] - out["hi"][1] > self.MR_MARGIN else "hi"
            else:
                lang = cur
            text, lp = out[lang]
            tried = len(out)
            if text:
                self.chosen = lang
        return {"text": text, "logprob": round(lp, 3), "lang": lang, "lang_p": None, "models_tried": tried,
                "ms": round((time.perf_counter() - t0) * 1000, 1)}

    def reset(self):
        self.chosen = None


def make_stt(spec, lang="auto3", threads=4):
    """spec like 'whisper:small' or 'whisper:small@10' (10 s chunk window, faster)."""
    if spec.startswith("indic"):
        _, _, chunk = spec.partition("@")
        return IndicSTT(threads=threads, chunk_s=int(chunk) if chunk else 10, lang=lang)
    kind, _, arg = spec.partition(":")
    if kind == "whisper":
        size, _, chunk = (arg or "small").partition("@")
        return WhisperSTT(size, lang=lang, threads=threads, chunk_s=int(chunk) if chunk else None)
    raise ValueError(f"unknown STT engine {spec!r}")
