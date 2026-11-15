"""State machine + safety layer. The only place that decides which script lines to say.

Safety rules:
  * only lines from script.json are ever said;
  * every failure / unknown ends in a keep-pushing line;
  * L10 (stop pushing) only in state `confirming`, only on a confident, non-negated,
    non-doubtful "breathing normally" (and, in voice mode, only if speech-to-text was
    confident too);
  * gasping is treated as not breathing.
"""
import json
import time
from pathlib import Path

from .nlu import NLU, detect_lang

ROOT = Path(__file__).resolve().parent.parent

START_CPR = ["L03", "L04", "L05"]


def load_script(path=None):
    return json.loads(Path(path or ROOT / "script.json").read_text(encoding="utf-8"))


class Dialog:
    STT_MIN_LOGPROB = -0.8   # voice mode: below this, a "stop" is not trusted

    def __init__(self, nlu=None, script=None, lang="hi", state="checking"):
        self.script = script or load_script()
        self.nlu = nlu or NLU()
        self.lang = lang
        self.state = state
        self.asked = None          # which checking question was asked last
        self.last_reply = []

    def start(self):
        self.state, self.asked = "checking", "respond"
        self.last_reply = ["L01"]
        return self.last_reply

    def line_text(self, lid, lang=None):
        line = self.script["lines"][lid]
        return line.get(lang or self.lang) or line["en"]

    # ------------------------------------------------------------------
    def handle(self, text, stt_logprob=None, state=None):
        t0 = time.perf_counter()
        if state:
            self.state = state if state in ("checking", "compressions", "confirming", "stopped") \
                else "compressions"
        try:
            a = self.nlu.analyze(text, self.state)
            if text and text.strip() and not getattr(self, "fixed_lang", False):
                self.lang = detect_lang(text, self.lang)
            reply, esc = self._decide(a, stt_logprob)
        except Exception as e:                       # never crash, fail safe
            a = {"intent": "unclear", "error": repr(e)}
            reply, esc = self._safe(), False
        if not reply:
            reply = self._safe()
        if "L10" in reply and self.state != "stopped":   # invariant guard
            reply, self.state = ["L13"], "compressions"
        self.last_reply = reply
        return {"intent": a["intent"], "reply": reply, "escalate": esc, "state": self.state,
                "lang": self.lang, "analysis": a, "ms": round((time.perf_counter() - t0) * 1000, 1)}

    def _safe(self):
        if self.state == "checking":
            self.state = "compressions"
            return list(START_CPR)
        self.state = "compressions"
        return ["L15"]

    def _go_cpr(self):
        self.state = "compressions"
        return list(START_CPR)

    def _decide(self, a, stt_logprob):
        i, s = a["intent"], self.state
        if i == "repeat":
            return (list(self.last_reply) or self._safe()), False
        if i == "out_of_scope":
            if s == "checking":
                self.state = "compressions"
            return ["L11"], True

        if s == "checking":
            if i == "not_responding":
                self.asked = "breathing"
                return ["L02"], False
            if i == "help_called":
                self.asked = "breathing"
                return ["L02"], False
            if i == "not_breathing":
                return self._go_cpr(), False
            if i == "gasping":
                return ["L12"] + self._go_cpr(), False
            if i in ("responding", "breathing_normal", "help_arrived", "tired"):
                # Not covered by the CPR script -> escalate, still keep-pushing wording.
                return ["L11"], True
            # where_hands, how_deep, pushing, unclear: when unsure, start CPR.
            return self._go_cpr(), False

        if s in ("compressions", "confirming", "stopped"):
            if s == "confirming":
                ok_stt = stt_logprob is None or stt_logprob >= self.STT_MIN_LOGPROB
                if i == "breathing_normal" and a.get("confident_stop") and ok_stt:
                    self.state = "stopped"
                    return ["L10"], False
                if i == "breathing_normal":           # not confident enough: ask again, keep pushing
                    return ["L15", "L09"], False
                self.state = "compressions"
                if i == "gasping":
                    return ["L12"], False
                if i == "unclear":
                    return ["L15"], False
                return ["L13"], False

            if s == "stopped":
                if i in ("not_breathing", "gasping", "not_responding"):
                    return self._go_cpr()[1:], False      # hands on chest, push
                return ["L11"], True

            # compressions
            table = {
                "tired": ["L08"], "where_hands": ["L14"], "how_deep": ["L06"], "pushing": ["L07"],
                "gasping": ["L12"], "not_breathing": ["L13"], "not_responding": ["L13"],
                "help_called": ["L13"], "help_arrived": ["L13"], "unclear": ["L15"],
            }
            if i in ("breathing_normal", "responding"):
                self.state = "confirming"
                return ["L09"], False
            return table.get(i, ["L15"]), False
        return self._safe(), False
