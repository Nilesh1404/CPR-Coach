"""Train + measure on your recorded clips.

  python train.py                          # transcribe clips, evaluate, print report
  python train.py --stt whisper:small,whisper:base   # compare STT engines
  python train.py --apply                  # also add clip transcripts to the classifier
  python train.py --typed-only             # no STT, use the typed transcripts

Evaluation is honest: "leave-one-speaker-out" - each speaker is tested with a classifier
trained on the seed examples + the OTHER speakers only.

Measured: critical-word error, intent accuracy (counts), unsafe replies, ms per stage, memory.
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from coach.dialog import Dialog, load_script  # noqa: E402
from coach.nlu import NLU, normalize, load_examples  # noqa: E402

# The words that change the instruction, with spellings that count as the same word.
CRITICAL = {
    "saans": ["saans", "sans", "saas", "सांस", "सास", "श्वास", "shwas", "swas"],
    "nahi": ["nahi", "nahin", "nai", "nhi", "नहीं", "नही", "नाही", "नाहीये"],
    "breathing": ["breathing", "breath", "ब्रीदिंग"],
    "not": ["not", "नॉट"],
    "थांबा": ["थांबा", "thamba", "थांब"],
    "हिल": ["हिल", "hil", "हल"],
}
CRIT_N = {k: {normalize(v) for v in vs} for k, vs in CRITICAL.items()}


def crit_words(text):
    toks = normalize(text).split()
    found = Counter()
    for k, vs in CRIT_N.items():
        for t in toks:
            if t in vs or any(t.startswith(v) and len(v) >= 3 for v in vs):
                found[k] += 1
    return found


def is_unsafe(true_intent, state, reply):
    """Unsafe = told to stop when not truly breathing, or a non-breathing person
    in the checking state did not get CPR started."""
    if "L10" in reply and true_intent != "breathing_normal":
        return True
    if state == "checking" and true_intent in ("not_breathing", "gasping") and "L05" not in reply:
        return True
    return False


def evaluate(rows, hyp_key, script, engine="knn"):
    speakers = sorted({r["speaker"] for r in rows})
    base_ex = load_examples_seed()
    res = []
    for spk in speakers:
        train_ex = base_ex + [{"text": t, "intent": r["intent"]}
                              for r in rows if r["speaker"] != spk
                              for t in (r.get(hyp_key), r.get("transcript")) if t]
        nlu = NLU(train_ex, engine=engine)
        for r in rows:
            if r["speaker"] != spk:
                continue
            d = Dialog(nlu=nlu, script=script)
            out = d.handle(r.get(hyp_key) or "", stt_logprob=r.get("logprob"), state=r["state"])
            res.append({**r, "pred": out["intent"], "reply": out["reply"], "nlu_ms": out["ms"],
                        "unsafe": is_unsafe(r["intent"], r["state"], out["reply"])})
    return res


def bad_hyp(text, intent):
    """Reason not to learn from this STT transcript, or None if it is OK.
    A transcript is only learned if it is clean and agrees with its label on the safety words."""
    from coach.nlu import has
    t = normalize(text)
    toks = t.split()
    if len(toks) < 1:
        return "empty"
    if len(toks) >= 6 and len(set(toks)) / len(toks) < 0.5:
        return "repetition loop"
    if not any("ऀ" <= ch <= "ॿ" for ch in text) and not any(
            w in toks for w in ("breathing", "breath", "tired", "moving", "push", "hands", "ambulance",
                                "not", "no", "yes", "help", "deep", "hard", "eyes", "normal")):
        return "English nonsense"
    neg = has(t, toks, "neg")
    if intent in ("breathing_normal", "responding", "pushing", "help_called", "help_arrived") and neg:
        return "has a 'not' word but label is positive"
    if intent in ("not_breathing", "not_responding") and not neg:
        return "label is 'not ...' but no 'not' word heard"
    return None


def load_examples_seed():
    p = ROOT / "data" / "examples.json"
    return json.loads(p.read_text(encoding="utf-8"))["examples"]


def report(name, res, rows_with_ref):
    n = len(res)
    ok = sum(r["pred"] == r["intent"] for r in res)
    unsafe = [r for r in res if r["unsafe"]]
    print(f"\n### {name}")
    print(f"intent accuracy: {ok}/{n}" + (f" ({100*ok/n:.1f}%)" if n else ""))
    print(f"unsafe replies : {len(unsafe)}/{n}")
    for r in unsafe:
        print(f"   UNSAFE {r['clip_id']}: said {r.get('hyp')!r} true={r['intent']} pred={r['pred']} -> {r['reply']}")
    by = defaultdict(lambda: [0, 0])
    for r in res:
        by[r["place"]][1] += 1
        by[r["place"]][0] += r["pred"] == r["intent"]
    print("by place: " + ", ".join(f"{p} {a}/{b}" for p, (a, b) in sorted(by.items())))
    conf = Counter((r["intent"], r["pred"]) for r in res if r["intent"] != r["pred"])
    if conf:
        print("top confusions (true -> pred):")
        for (t, p), c in conf.most_common(8):
            print(f"   {c} x {t} -> {p}")
    # critical words: compare typed transcript (reference) vs STT output
    ref_tot, dels, ins = Counter(), Counter(), Counter()
    for r in rows_with_ref:
        if not r.get("transcript") or "hyp" not in r:
            continue
        ref, hyp = crit_words(r["transcript"]), crit_words(r["hyp"])
        for k in CRITICAL:
            ref_tot[k] += ref[k]
            dels[k] += max(0, ref[k] - hyp[k])
            ins[k] += max(0, hyp[k] - ref[k])
    if sum(ref_tot.values()):
        print("critical-word errors (missed / extra / in reference):")
        for k in CRITICAL:
            if ref_tot[k] or ins[k]:
                print(f"   {k:10s} missed {dels[k]:3d}  extra {ins[k]:3d}  of {ref_tot[k]}")
    ms = [r["stt_ms"] for r in rows_with_ref if r.get("stt_ms")]
    nl = [r["nlu_ms"] for r in res]
    if ms:
        print(f"STT ms: median {sorted(ms)[len(ms)//2]:.0f}, max {max(ms):.0f}")
    if nl:
        print(f"NLU ms: median {sorted(nl)[len(nl)//2]:.1f}, max {max(nl):.1f}")
    return {"name": name, "n": n, "correct": ok, "unsafe": len(unsafe),
            "by_place": dict(by), "confusions": [[t, p, c] for (t, p), c in conf.most_common()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="clips/labels.csv")
    ap.add_argument("--stt", default="whisper:small", help="comma-separated engines to compare")
    ap.add_argument("--lang", default="auto3")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--typed-only", action="store_true")
    ap.add_argument("--apply", action="store_true", help="write data/learned_examples.json")
    ap.add_argument("--nlu", default="knn", help="line choosers to compare: knn (char n-grams), embed (multilingual MiniLM), e.g. knn,embed")
    args = ap.parse_args()

    # utf-8-sig: labels.csv may start with a BOM (saved for Excel)
    rows = list(csv.DictReader(open(ROOT / args.labels, encoding="utf-8-sig")))
    known = set(load_script()["intents"])
    skipped = [r["clip_id"] for r in rows if r["intent"] not in known]
    rows = [r for r in rows if r["intent"] in known]
    if skipped:
        print(f"skipped {len(skipped)} clips with unknown intent (fill them in labels.csv): "
              + ", ".join(skipped))
    print(f"{len(rows)} clips, {len({r['speaker'] for r in rows})} speakers, "
          f"places: {dict(Counter(r['place'] for r in rows))}")
    script = load_script()
    reports = []
    cache_p = ROOT / "clips" / "stt_cache.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}

    if args.typed_only:
        for r in rows:
            r["hyp"] = r["transcript"]
        for eng in args.nlu.split(","):
            reports.append(report(f"typed transcripts (no STT) + {eng}", evaluate(rows, "hyp", script, eng), []))
    else:
        import psutil
        from coach.audio import load_wav
        from coach.stt import make_stt
        proc = psutil.Process()
        for spec in args.stt.split(","):
            rss0 = proc.memory_info().rss
            t0 = time.perf_counter()
            stt = make_stt(spec, lang=args.lang, threads=args.threads)
            load_s = time.perf_counter() - t0
            peak = proc.memory_info().rss
            hint = {}   # indic: like a real call, language carries over per speaker, starting at Hindi
            for r in rows:
                key = f"{spec}|{args.lang}|{r['file']}"
                if key not in cache:
                    if spec.startswith("indic"):
                        if hint.get("_spk") != r["speaker"]:      # new speaker = new call
                            stt.reset()
                            hint["_spk"] = r["speaker"]
                        s = stt.transcribe(load_wav(ROOT / r["file"]))
                    else:
                        s = stt.transcribe(load_wav(ROOT / r["file"]), sticky=False)
                    cache[key] = s
                    print(f"  {r['clip_id']}: {s['text']!r} ({s['ms']:.0f} ms)")
                    peak = max(peak, proc.memory_info().rss)
                s = cache[key]
                r["hyp"], r["logprob"], r["stt_ms"] = s["text"], s["logprob"], s["ms"]
            cache_p.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"\n{spec}: load {load_s:.1f}s, memory +{(peak - rss0) / 2**20:.0f} MB "
                  f"(process {peak / 2**20:.0f} MB)")
            for eng in args.nlu.split(","):
                rss1 = proc.memory_info().rss
                rep = report(f"{spec} + {eng}/rules", evaluate(rows, "hyp", script, eng), rows)
                rep["load_s"], rep["mem_mb"] = round(load_s, 1), round((peak - rss0) / 2**20)
                rep["nlu_mem_mb"] = round((proc.memory_info().rss - rss1) / 2**20)
                print(f"{eng}: extra memory for line chooser +{rep['nlu_mem_mb']} MB")
                reports.append(rep)
            del stt

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    rp = out / f"train_{time.strftime('%Y%m%d-%H%M%S')}.json"
    rp.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport -> {rp}")

    if args.apply:
        ex, dropped = [], []
        for r in rows:
            if r.get("transcript"):
                ex.append({"text": r["transcript"], "intent": r["intent"], "src": r["clip_id"]})
            h = r.get("hyp")
            if h and h != r.get("transcript"):
                why = bad_hyp(h, r["intent"])
                if why:
                    dropped.append((r["clip_id"], why, h))
                else:
                    ex.append({"text": h, "intent": r["intent"], "src": r["clip_id"] + ":stt"})
        print(f"dropped {len(dropped)} STT transcripts that would teach the wrong thing:")
        for cid, why, h in dropped:
            print(f"   {cid}: {why}: {h[:60]!r}")
        p = ROOT / "data" / "learned_examples.json"
        p.write_text(json.dumps({"examples": ex}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"added {len(ex)} learned examples -> {p}  (used by run.py from now on)")


if __name__ == "__main__":
    main()
