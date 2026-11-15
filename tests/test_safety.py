"""Safety checks: python tests/test_safety.py  (exit code 1 on any failure)."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coach.dialog import Dialog  # noqa: E402
from coach.nlu import NLU  # noqa: E402

nlu = NLU()
fails = 0


def reply(text, state):
    return Dialog(nlu=nlu).handle(text, state=state)


def check(cond, msg):
    global fails
    if not cond:
        fails += 1
        print("FAIL", msg)


# 1. negated / gasping / doubtful sentences must never stop compressions
must_not_stop = [
    "saans nahi le raha", "nahi", "no", "not breathing", "breathing nahi kar raha",
    "saans nahi aa rahi", "nahin saans nahi", "श्वास घेत नाही", "नाही", "सांस नहीं",
    "haanf raha hai", "gasping", "ajeeb saans le raha", "ruk ruk ke saans", "shayad saans le raha",
    "lagta hai saans le raha", "pata nahi", "maybe breathing", "i think he is breathing",
    "haan nahi", "haan saans nahi le raha", "kabhi kabhi saans", "", "hmm", "kya", "blah",
    "थोडा थोडा श्वास", "no he is not", "धापा टाकतोय", "हाँफ रहा है", "not sure",
]
for t, st in itertools.product(must_not_stop, ["checking", "compressions", "confirming"]):
    r = reply(t, st)
    check("L10" not in r["reply"], f"stopped on {t!r} in {st}: {r['reply']}")

# 2. L10 only ever in confirming
for t in ["haan saans le raha hai", "yes he is breathing normally", "haan", "हो श्वास घेत आहे"]:
    for st in ["checking", "compressions"]:
        check("L10" not in reply(t, st)["reply"], f"L10 outside confirming: {t!r} {st}")
    check(reply(t, "confirming")["reply"] == ["L10"], f"clear yes not accepted: {t!r}")

# 3. low STT confidence never stops
d = Dialog(nlu=nlu)
check("L10" not in d.handle("haan saans le raha", stt_logprob=-1.5, state="confirming")["reply"],
      "stopped on low STT confidence")

# 4. every non-stop reply in compressions/confirming keeps pushing
keep = {"L05", "L06", "L07", "L08", "L09", "L11", "L12", "L13", "L14", "L15"}
for t in must_not_stop + ["paani dein?", "haath kahan", "thak gaya", "ambulance aa gayi"]:
    for st in ["compressions", "confirming"]:
        r = reply(t, st)
        check(set(r["reply"]) <= keep, f"non keep-pushing reply {r['reply']} for {t!r} {st}")

# 5. checking + not breathing/unclear -> CPR starts
for t in ["saans nahi le raha", "gasping", "asdf qwer", ""]:
    check("L05" in reply(t, "checking")["reply"], f"CPR not started for {t!r}")

print("all safety checks passed" if not fails else f"{fails} failures")
sys.exit(1 if fails else 0)
