"""Offline CPR voice coach.

  python run.py --text utterances.json results.json   # batch text test
  python run.py --chat [--beat]                        # type sentences, see replies
  python run.py --voice [--ptt] [--stt whisper:small@10]  # the real thing, mic + speaker
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from coach.dialog import Dialog, START_CPR, load_script  # noqa: E402
from coach.nlu import NLU  # noqa: E402


def text_mode(inp, outp, engine="knn"):
    nlu, script = NLU(engine=engine), load_script()
    items = json.loads(Path(inp).read_text(encoding="utf-8")).get("items", [])
    results = []
    for it in items:
        t0 = time.perf_counter()
        try:
            d = Dialog(nlu=nlu, script=script, lang="hi")
            r = d.handle(it.get("text", ""), state=it.get("state", "compressions"))
            res = {"id": it.get("id"), "intent": r["intent"], "reply": r["reply"],
                   "escalate": r["escalate"]}
        except Exception as e:  # every item must get a result
            st = it.get("state")
            res = {"id": it.get("id"), "intent": "unclear",
                   "reply": START_CPR if st == "checking" else ["L15"],
                   "escalate": False, "error": repr(e)}
        res["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        results.append(res)
    Path(outp).write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"wrote {len(results)} results to {outp}")


def show(d, r, t_stt=None):
    a = r["analysis"]
    extra = f" stt={t_stt:.0f}ms" if t_stt is not None else ""
    print(f"  -> intent={r['intent']} score={a.get('score')} state={r['state']} "
          f"lang={r['lang']} esc={r['escalate']} nlu={r['ms']}ms{extra}")
    for lid in r["reply"]:
        print(f"  [{lid}] {d.line_text(lid)}")


def chat_mode(args):
    d = Dialog(nlu=NLU(engine=args.nlu), lang=args.lang_reply)
    player = voices = None
    if args.beat:
        from coach.audio import Player, Voices
        player, voices = Player(), Voices(",".join(str(ROOT / v) for v in args.voices.split(",")))
        player.start()
    first = d.start()
    print("Type what the bystander says (empty line = silence, 'q' = quit).")
    say(d, first, player, voices)
    while True:
        try:
            s = input(f"[{d.state}] you> ")
        except EOFError:
            break
        if s.strip().lower() == "q":
            break
        r = d.handle(s)
        if player:
            player.beat_on = d.state in ("compressions", "confirming")
        show(d, r)
        say(d, r["reply"], player, voices, quiet=True)


def say(d, ids, player, voices, quiet=False):
    if not quiet:
        for lid in ids:
            print(f"  [{lid}] {d.line_text(lid)}")
    if player is None:
        return
    for lid in ids:
        audio, used = voices.get(lid, d.lang)
        if used is None:
            print(f"     (no recording for {lid}/{d.lang}; chime)")
        player.play(audio)


def voice_mode(args):
    from coach.audio import Mic, Player, Voices, save_wav
    from coach.stt import make_stt
    import psutil

    print(f"loading {args.stt} ...")
    stt = make_stt(args.stt, lang=args.lang, threads=args.threads)
    d = Dialog(nlu=NLU(engine=args.nlu), lang=args.lang_reply)
    player, mic, voices = Player(beat_vol=args.beat_vol), Mic(), Voices(",".join(str(ROOT / v) for v in args.voices.split(",")))
    player.start()
    mic.start()
    logdir = ROOT / "logs" / time.strftime("%Y%m%d-%H%M%S")
    logdir.mkdir(parents=True, exist_ok=True)
    log = open(logdir / "session.jsonl", "a", encoding="utf-8")
    proc = psutil.Process()

    say(d, d.start(), player, voices)
    t_session = time.time()
    n = 0
    print("Listening. Ctrl+C to quit.")
    try:
        while True:
            if args.ptt:
                audio = mic.listen_ptt()
            else:
                audio = mic.listen_vad(player, half_duplex=not args.full_duplex,
                                       duck=args.duck)
            n += 1
            s = stt.transcribe(audio, hint=d.lang)
            r = d.handle(s["text"], stt_logprob=s["logprob"])
            player.beat_on = d.state in ("compressions", "confirming")
            print(f"\nheard: {s['text']!r} (logprob {s['logprob']})")
            show(d, r, s["ms"])
            say(d, r["reply"], player, voices, quiet=True)
            if args.save_audio:
                save_wav(logdir / f"u{n:03d}.wav", audio)
            log.write(json.dumps({
                "n": n, "t": round(time.time() - t_session, 2), "audio_s": round(len(audio) / 16000, 2),
                "stt": s, "intent": r["intent"], "reply": r["reply"], "state": r["state"],
                "escalate": r["escalate"], "nlu_ms": r["ms"], "score": r["analysis"].get("score"),
                "rss_mb": round(proc.memory_info().rss / 2**20), "beats": player.beats_played,
            }, ensure_ascii=False) + "\n")
            log.flush()
    except KeyboardInterrupt:
        print(f"\nbye. log: {logdir}")
    finally:
        player.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--voice", action="store_true")
    ap.add_argument("--beat", action="store_true", help="chat mode: also play beat + voices")
    ap.add_argument("--stt", default="whisper:small@10")
    ap.add_argument("--lang", default="auto3", help="STT language: auto3 (best of hi/mr/en), hi, mr, en, auto")
    ap.add_argument("--lang-reply", default="hi", help="starting reply language")
    ap.add_argument("--nlu", default="knn", choices=["knn", "embed"], help="line chooser")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--voices", default="voices/human,voices/female,voices/synth",
                    help="comma-separated folders with <lang>/<Lxx>.wav, tried in order")
    ap.add_argument("--ptt", action="store_true", help="push-to-talk instead of auto VAD")
    ap.add_argument("--full-duplex", action="store_true", help="listen while speaking")
    ap.add_argument("--duck", type=float, default=None, help="beat volume factor while listening")
    ap.add_argument("--beat-vol", type=float, default=0.35)
    ap.add_argument("--save-audio", action="store_true", help="keep utterance WAVs (consent!)")
    args = ap.parse_args()
    if args.text:
        text_mode(*args.text, engine=args.nlu)
    elif args.chat:
        chat_mode(args)
    else:
        voice_mode(args)


if __name__ == "__main__":
    main()
