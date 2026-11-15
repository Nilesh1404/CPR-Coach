"""Understanding: text -> intent.

Two parts:
  1. A small nearest-neighbour classifier over character n-grams. It learns from
     examples (data/examples.json + data/learned_examples.json written by train.py).
     Character n-grams cope with romanised Hindi/Marathi spelling variation
     ("saans", "sans", "saas") and with Devanagari.
  2. Hard keyword rules for the words that flip the instruction (negation,
     gasping, doubt). These override the classifier, because a single "nahi"
     must never be outvoted by similarity scores.
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def normalize(text):
    t = unicodedata.normalize("NFC", text or "").lower()
    t = t.replace("ँ", "ं")          # chandrabindu -> anusvara (साँस == सांस)
    t = t.replace("़", "")                # drop nukta (ज़ == ज)
    t = re.sub(r"[।॥]", " ", t)      # danda
    t = t.replace("'", "").replace("’", "")
    t = re.sub(r"[^\w\sऀ-ॿ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# Patterns: plain = whole token, "x*" = token prefix, "a b" = phrase.
LEX = {
    "neg": ["nahi", "nahin", "nahee", "nai", "nhi", "nahiye", "nahiyay", "नहीं", "नही", "नाही", "नाहीये",
            "नहीं*", "नाही*", "not", "no", "nope", "isnt", "doesnt", "dont", "cant", "नॉट", "नो",
            "band", "बंद", "ruk gayi", "रुक गई", "थांबला", "थांबलाय",
            # Whisper merges "घेत नाहीये" into one word: "गेतने ये", "घेतनाही"
            "घेतन*", "गेतन*", "घेतना*", "गेतना*", "गितना*", "गीतना*", "ना ये", "नाये", "नाहिये", "नहिये", "नगे"],
    "doubt": ["shayad", "शायद", "maybe", "lagta", "लगता", "pata", "पता", "think", "sure",
              "kadachit", "कदाचित", "vatat", "वाटत*", "kya pata", "malum", "मालूम", "thoda thoda", "थोडा थोडा"],
    "gasp": ["haanf*", "hanf*", "hamf*", "gasp*", "हांफ*", "हाफ*", "धाप*", "ukhd*", "ukhad*", "उखड*",
             "ghargh*", "घरघर*", "अजिव*", "अजीव*", "अजिब*", "आजिब*", "अजब*", "धापा*", "दापा*", "ruk ruk", "रुक रुक", "kabhi kabhi", "कभी कभी", "ajeeb", "अजीब",
             "khar khar", "snor*", "jhatke", "झटके"],
    "breath": ["saans", "sans", "saas", "saans*", "सांस*", "सास", "श्वास*", "shwas*", "swas*", "shvas*",
               "breath*", "ब्रीद*", "ब्रेथ*", "दम"],
    "yes": ["haan", "han", "haa", "ha", "ho", "hoy", "haanji", "yes", "yeah", "yep", "ji",
            "हां", "हा", "हो", "होय", "जी", "हांजी", "येस"],
    "no": ["nahi", "nahin", "nai", "no", "नहीं", "नही", "नाही", "नो"],
    # Positive evidence needed before L10 (stop): a yes word or a "is breathing" verb.
    # A garbled transcript with only "साँस" in it must not stop CPR.
    "affirm": ["haan", "han", "haa", "ho", "hoy", "yes", "yeah", "हां", "हा", "हो", "होय", "जी", "हांजी", "येस",
               "ले रहा", "ले रही", "ले रहे", "le raha", "le rahi", "le rahe", "चल रही", "chal rahi",
               "आ रही", "aa rahi", "घेतोय", "घेतेय", "घेतोया", "घेत आहे", "घेत आहेत", "ghetoy", "ghetey",
               "ghet ahe", "ghet aahe", "normal", "नॉर्मल", "नार्मल", "नारमल", "breathing", "ब्रीदिंग"],
}


def _compile(pats):
    out = []
    for p in pats:
        star = p.endswith("*")
        p = normalize(p.rstrip("*"))
        out.append((p, star))
    return out


LEXC = {k: _compile(v) for k, v in LEX.items()}


def has(norm_text, tokens, kind):
    padded = f" {norm_text} "
    for p, star in LEXC[kind]:
        if " " in p:
            if f" {p} " in padded:
                return True
        elif star:
            if any(t.startswith(p) for t in tokens):
                return True
        elif p in tokens:
            return True
    return False


# ---------------- language detection (for choosing the reply language) ----
# Whisper's language ID cannot tell Marathi from Hindi (Marathi speech scores ~0.9 "hi"), and it
# then writes Marathi in Hindi-ish spelling. So the reply language is decided here from the words.
# Patterns: "x" whole token, "x*" prefix, "*x" suffix (verb endings).
MR = ["आहे", "आहेत", "आहेस", "आहोत", "नाही", "नाहीये", "नाहीत", "नाहीयेत", "घेत*", "करा", "करू", "करु", "झाल*",
      "झालं", "काय", "मी", "मला", "मलाच", "आम्ही", "तुम्ही", "तू", "त्याला", "तिला", "त्यांना", "त्याने", "तिने",
      "थकलो", "थकले", "थकली", "थकवा", "होय", "ती", "कुठे", "कुठं", "किती", "आता", "आत्ता", "परत",
      "पुन्हा", "सांग*", "हलत*", "देत", "आली", "आला", "केला", "केली", "ठेवू", "ठेव*", "दाबू", "दाबायच*",
      "कळत", "कळलं", "समजत", "समजलं", "माहित", "माहीत", "उघड*", "डोळे", "बोलत", "उठत", "येत", "जात", "आणि",
      "पण", "नको", "बरं", "छान", "अजून", "खूप", "जोरात", "हात", "काहीच", "कसं", "कसा", "कशी",
      "*तोय", "*तेय", "*लाय", "*लंय", "*लेय", "*ायचं", "*ायचा", "*ायची", "*तात", "*तोस", "*लोय",
      "ahe", "aahe", "ahet", "aahet", "ghet*", "kara", "karu", "zala", "jhala", "zal*", "thaklo", "thakle",
      "kay", "nahiye", "nahit", "hoy", "kuthe", "kiti", "ata", "aata", "aali", "aala", "kela", "dabu", "sang*",
      "mala", "tyala", "tila", "tumhi", "aamhi", "punha", "parat", "kalat", "samjat", "mahit", "bolat",
      "uthat", "ani", "aani", "nako", "jorat", "haat", "kahich", "kasa", "kashi", "thevu", "*toy", "*laay"]
HI = ["है", "हैं", "हूं", "था", "थी", "थे", "नहीं", "नही", "रहा", "रही", "रहे", "क्या", "मैं", "मैंने",
      "मुझे", "हम", "आप", "उसने", "उसको", "उसे", "उन्होंने", "गया", "गए", "गई", "कहां", "कितना", "कितनी",
      "कैसे", "अभी", "किया", "दिया", "रखूं", "रखना", "करूं", "करना", "दबाऊं", "दबाना", "फिर", "से", "को",
      "में", "पर", "और", "लेकिन", "बोलो", "बताओ", "बताइए", "समझ", "आंखें", "खोली", "हिल*", "हाथ", "ज़ोर", "जोर",
      "थक", "आ", "वो", "वह", "यह", "ये", "कुछ", "भी", "हां", "जी", "हांजी", "बहुत", "ठीक", "चाहिए",
      "hai", "hain", "hoon", "tha", "thi", "raha", "rahi", "rahe", "gaye", "gaya", "gayi", "kya", "nahin",
      "main", "maine", "mujhe", "kahan", "kitna", "kaise", "abhi", "kiya", "diya", "bhai", "rakhu", "phir",
      "fir", "bolo", "batao", "samajh", "haan", "haath", "zor", "thak", "wo", "woh", "yeh", "kuch", "bhi",
      "bahut", "theek", "thik", "chahiye", "unhone", "usne", "aankhen", "khol*"]
EN = ["is", "not", "the", "he", "she", "breathing", "tired", "where", "yes", "how", "i", "am", "my",
      "hands", "push", "help", "they", "what", "here", "called", "again", "no", "it", "are", "do", "does",
      "should", "can", "you", "we", "has", "have", "now", "ok", "okay", "please", "repeat", "eyes",
      "moving", "came", "arrived", "ambulance", "deep", "hard", "fast", "stop", "come", "put", "was"]
_DEV = re.compile("[ऀ-ॿ]")


def _compile_lang(pats):
    out = []
    for p in pats:
        kind = "suf" if p.startswith("*") else "pre" if p.endswith("*") else "tok"
        out.append((normalize(p.strip("*")), kind))
    return out


LANGC = {"mr": _compile_lang(MR), "hi": _compile_lang(HI), "en": _compile_lang(EN)}


def lang_scores(text):
    toks = normalize(text).split()
    scores = {}
    for lang, pats in LANGC.items():
        s = 0
        for tok in toks:
            for p, kind in pats:
                if (kind == "tok" and tok == p) or (kind == "pre" and tok.startswith(p)) or                         (kind == "suf" and len(tok) > len(p) + 1 and tok.endswith(p)):
                    s += 1
                    break
        scores[lang] = s
    return scores, toks


def detect_lang(text, default="hi"):
    scores, toks = lang_scores(text)
    if not toks:
        return default
    dev = bool(_DEV.search(text or ""))
    if dev:                                     # Devanagari is Hindi or Marathi, never English
        scores["en"] = -1
        if default == "en":
            default = "hi"
    top = max(scores.values())
    if top <= 0:
        return default
    tied = [l for l in ("mr", "hi", "en") if scores[l] == top]
    # Tie: a Marathi-only word is the stronger signal (Whisper spells Marathi like Hindi,
    # e.g. नाही -> नहीं), and any Hindi/Marathi word beats English filler.
    return tied[0]


# ---------------- classifier ------------------------------------------------
def features(t):
    s = f" {t} "
    c = Counter()
    for n in (2, 3, 4):
        for i in range(len(s) - n + 1):
            c["c" + s[i:i + n]] += 1
    for w in t.split():
        c["w" + w] += 2
    return c


class KNN:
    def __init__(self, examples):
        self.texts = [normalize(e["text"]) for e in examples]
        self.labels = [e["intent"] for e in examples]
        feats = [features(t) for t in self.texts]
        vocab = {}
        df = Counter()
        for f in feats:
            df.update(f.keys())
            for k in f:
                vocab.setdefault(k, len(vocab))
        self.vocab = vocab
        n = len(feats)
        self.idf = np.zeros(len(vocab), np.float32)
        for k, i in vocab.items():
            self.idf[i] = np.log((1 + n) / (1 + df[k])) + 1
        self.M = np.vstack([self._vec(f) for f in feats]) if feats else np.zeros((0, len(vocab)))

    def _vec(self, f):
        v = np.zeros(len(self.vocab), np.float32)
        for k, c in f.items():
            i = self.vocab.get(k)
            if i is not None:
                v[i] = (1 + np.log(c)) * self.idf[i]
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    def predict(self, t):
        if not len(self.labels):
            return "unclear", 0.0, None
        sims = self.M @ self._vec(features(t))
        best = {}
        for i, s in enumerate(sims):
            lab = self.labels[i]
            if s > best.get(lab, (-1, None))[0]:
                best[lab] = (float(s), self.texts[i])
        lab = max(best, key=lambda k: best[k][0])
        return lab, best[lab][0], best[lab][1]


class EmbedKNN:
    """Option 2 for choosing the line: multilingual sentence embeddings
    (paraphrase-multilingual-MiniLM-L12-v2, ~220 MB, ONNX via fastembed, CPU, offline after setup).
    Compares meaning instead of spelling. Weak on negation ("saans le raha" ~ "saans nahi le raha"),
    so the same keyword safety rules in NLU.analyze still decide negation / gasping."""
    MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    CACHE = ROOT / "models" / "fastembed"
    _model = None
    _memo = {}

    @classmethod
    def model(cls):
        if cls._model is None:
            from fastembed import TextEmbedding
            cls._model = TextEmbedding(cls.MODEL, cache_dir=str(cls.CACHE))
        return cls._model

    @classmethod
    def embed(cls, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in cls._memo]
        if todo:
            for t, v in zip(todo, cls.model().embed(todo)):
                v = np.asarray(v, np.float32)
                cls._memo[t] = v / (np.linalg.norm(v) or 1)
        return np.vstack([cls._memo[t] for t in texts]) if texts else np.zeros((0, 384), np.float32)

    def __init__(self, examples):
        self.texts = [normalize(e["text"]) for e in examples]
        self.labels = [e["intent"] for e in examples]
        self.M = self.embed(self.texts)

    def predict(self, t):
        if not len(self.labels):
            return "unclear", 0.0, None
        sims = self.M @ self.embed([t])[0]
        i = int(np.argmax(sims))
        return self.labels[i], float(sims[i]), self.texts[i]


def load_examples():
    ex = json.loads((ROOT / "data" / "examples.json").read_text(encoding="utf-8"))["examples"]
    learned = ROOT / "data" / "learned_examples.json"
    if learned.exists():
        ex += json.loads(learned.read_text(encoding="utf-8"))["examples"]
    return ex


class NLU:
    MIN_SCORE = 0.30      # below this the classifier is "unclear"
    STOP_SCORE = 0.45     # classifier confidence needed for a stop without keyword support

    # per line-chooser: (MIN_SCORE, STOP_SCORE); embedding similarities run higher than n-gram ones
    THRESHOLDS = {"knn": (0.30, 0.45), "embed": (0.60, 0.80)}

    def __init__(self, examples=None, engine="knn"):
        ex = examples if examples is not None else load_examples()
        self.engine = engine
        self.knn = EmbedKNN(ex) if engine == "embed" else KNN(ex)
        self.MIN_SCORE, self.STOP_SCORE = self.THRESHOLDS[engine]

    def analyze(self, text, state):
        t = normalize(text)
        toks = t.split()
        out = {"text": text, "norm": t, "intent": "unclear", "score": 0.0, "nearest": None}
        if not toks:
            out.update(flags={}, confident_stop=False)
            return out
        f = {k: has(t, toks, k) for k in ("neg", "doubt", "gasp", "breath", "affirm")}
        f["yes_first"] = has(toks[0], toks[:1], "yes")
        f["no_first"] = has(toks[0], toks[:1], "no")

        intent, score, nearest = self.knn.predict(t)
        if score < self.MIN_SCORE:
            intent = "unclear"
        out.update(score=round(score, 3), nearest=nearest, knn_intent=intent)

        # ---- safety overrides (rules beat the classifier) ----
        breath_related = {"breathing_normal", "not_breathing", "gasping", "unclear",
                          "responding", "not_responding", "pushing"}
        if f["gasp"]:
            intent = "gasping"
        elif f["breath"] and intent in breath_related:
            intent = "not_breathing" if f["neg"] else "breathing_normal"
        elif intent == "breathing_normal" and f["neg"]:
            intent = "not_breathing"
        elif intent == "responding" and f["neg"]:
            intent = "not_responding"

        # In the confirming state the question is "yes or no?"
        if state == "confirming" and len(toks) <= 4 and not f["gasp"]:
            if f["no_first"] or (f["neg"] and intent in ("pushing", "unclear", "responding")):
                intent = "not_breathing"
            elif f["yes_first"] and not f["neg"] and intent in ("pushing", "unclear", "responding",
                                                                 "breathing_normal"):
                intent = "breathing_normal"

        out["intent"] = intent
        out["flags"] = f
        out["confident_stop"] = (
            intent == "breathing_normal" and not f["neg"] and not f["doubt"] and not f["gasp"]
            and f["affirm"]
            and (f["breath"] or f["yes_first"] or score >= self.STOP_SCORE)
        )
        return out
