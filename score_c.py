"""Score the "coach hearing itself" test from voice-session logs.

Say the 10 sentences of docs/C_test_sheet.docx in order, once per setup, each in its own
`python run.py --voice ...` session. Then:

  python score_c.py logs/<speaker-off run> logs/<beat-on run> logs/<fix run>

For each run it counts: intents right, key words kept, "nahi" missed, unsafe replies.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from coach.nlu import normalize  # noqa: E402

# (sentence, expected intent, key words that must survive, "nahi" said?)
SHEET = [
    ("साँस नहीं ले रहा", "not_breathing", ["saans", "nahi"], True),
    ("श्वास घेत नाही", "not_breathing", ["saans", "nahi"], True),
    ("breathing nahi kar raha", "not_breathing", ["breathing", "nahi"], True),
    ("he is not breathing", "not_breathing", ["not", "breathing"], False),
    ("हो, श्वास घेतोय", "breathing_normal", ["saans"], False),
    ("अजीब साँस, हाँफ रहा है", "gasping", ["saans"], False),
    ("वो हिल रहा है", "responding", ["hil"], False),
    ("थांबू का?", "keep_pushing", ["thamba"], False),
    ("हात दुखतायत, मी थकलो", "tired", [], False),
    ("हात कुठे ठेवू?", "where_hands", [], False),
]
KEYS = {
    "saans": ["saans", "sans", "saas", "सांस", "सास", "श्वास", "स्वास", "shwas", "swas"],
    "nahi": ["nahi", "nahin", "nai", "नहीं", "नही", "नाही", "नाहीये", "नाहीं"],
    "breathing": ["breathing", "breath", "ब्रीदिंग", "ब्रिथिंग", "ब्रिदिंग", "ब्रेथिंग", "ब्रिटिंग", "ब्रेडिंग", "ब्रीडिंग", "ब्रिडिंग"],
    "not": ["not", "नॉट"],
    "thamba": ["थांबा", "थांबू", "थांब", "thamba", "thambu"],
    "hil": ["हिल", "हल", "hil", "hal"],
}


def has_key(text, key):
    toks = normalize(text).split()
    vs = [normalize(v) for v in KEYS[key]]
    return any(t == v or (len(v) >= 3 and t.startswith(v)) for t in toks for v in vs)


def score(run_dir):
    p = Path(run_dir)
    p = p / "session.jsonl" if p.is_dir() else p
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    right = kept = keys_total = nahi_missed = unsafe = 0
    lines = []
    for i, (sent, want, keys, said_nahi) in enumerate(SHEET):
        if i >= len(rows):
            lines.append(f"  {i + 1:2}. {sent}  -> (not said / not logged)")
            continue
        r = rows[i]
        heard, intent, reply = r["stt"].get("text", ""), r["intent"], r["reply"]
        ok = ("L10" not in reply) if want == "keep_pushing" else intent == want
        right += ok
        k = sum(has_key(heard, key) for key in keys)
        kept += k
        keys_total += len(keys)
        if said_nahi and not has_key(heard, "nahi"):
            nahi_missed += 1
        bad = "L10" in reply and want != "breathing_normal"
        unsafe += bad
        lines.append(f"  {i + 1:2}. {sent:26} heard {heard!r:40} {intent:16} {' '.join(reply):12}"
                     f"{'  OK' if ok else '  WRONG'}{'  UNSAFE' if bad else ''}")
    extra = len(rows) - len(SHEET)
    return right, kept, keys_total, nahi_missed, unsafe, lines, extra


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    names = ["Speaker off", "Beat + voice on, mic always open", "Fix"]
    summary = []
    for n, run in enumerate(sys.argv[1:]):
        right, kept, total, nahi, unsafe, lines, extra = score(run)
        name = names[n] if n < len(names) else run
        print(f"\n== {name}  ({run})")
        print("\n".join(lines))
        if extra > 0:
            print(f"  note: {extra} extra utterance(s) logged (noise or repeats); only the first 10 are scored")
        summary.append((name, right, kept, total, nahi, unsafe))
    print("\n| Setup | Intents right | Key words kept | \"nahi\" missed | Unsafe |")
    print("| --- | --- | --- | --- | --- |")
    for name, right, kept, total, nahi, unsafe in summary:
        print(f"| {name} | {right} / {len(SHEET)} | {kept} / {total} | {nahi} / 3 | {unsafe} |")


if __name__ == "__main__":
    main()
