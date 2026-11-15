"""Record labelled training/test clips from one speaker.

  python record_clips.py --speaker S01 --gender f --age 34 --lang mr --place junction --consent yes

For each prompt: the speaker says it IN THEIR OWN WORDS, as a frightened bystander.
Enter = start, Enter = stop. Then type what they actually said (or leave blank and fill
in labels.csv later). Clips are saved to clips/<speaker>/ and listed in clips/labels.csv.
Nothing is uploaded anywhere.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402

from coach.audio import SR, Mic, save_wav  # noqa: E402

# (intent, state, what to act out). Negation pairs come first and twice - they matter most.
PROMPTS = [
    ("not_breathing", "checking", "Say the person is NOT breathing (e.g. 'saans nahi le raha')"),
    ("breathing_normal", "confirming", "Say YES, the person IS breathing normally now"),
    ("gasping", "compressions", "Say the person is gasping / making strange breathing sounds"),
    ("not_responding", "checking", "Say the person is NOT responding / not moving"),
    ("responding", "compressions", "Say the person moved / opened eyes / is responding"),
    ("not_breathing", "confirming", "Answer NO, still not breathing"),
    ("breathing_normal", "compressions", "Say 'he is breathing' in your own words"),
    ("tired", "compressions", "Say your arms are tired, you can't continue"),
    ("where_hands", "compressions", "Ask where to put your hands"),
    ("how_deep", "compressions", "Ask how hard / how deep to push"),
    ("help_called", "checking", "Say you have called the ambulance / 108"),
    ("help_arrived", "compressions", "Say the ambulance / doctor has arrived"),
    ("pushing", "compressions", "Say OK, you are pushing now"),
    ("repeat", "compressions", "Say you didn't understand, ask to repeat"),
    ("out_of_scope", "compressions", "Ask something else (water? bleeding? lift him?)"),
]
FIELDS = ["clip_id", "file", "speaker", "gender", "age", "lang", "place", "intent", "state",
          "transcript", "recorded_at"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--gender", default="")
    ap.add_argument("--age", default="")
    ap.add_argument("--lang", required=True, help="hi, mr, en, mix")
    ap.add_argument("--place", required=True, help="quiet, junction, station, market, ...")
    ap.add_argument("--consent", required=True, choices=["yes"],
                    help="speaker signed the consent form (required)")
    ap.add_argument("--rounds", type=int, default=1, help="times to go through the prompt list")
    ap.add_argument("--intents", default="",
                    help="only these prompts, in order, e.g. not_breathing@checking,gasping,tired "
                         "(see docs/recording plan)")
    ap.add_argument("--device", default=None, help="input device index/name")
    args = ap.parse_args()

    clips = ROOT / "clips"
    clips.mkdir(exist_ok=True)
    labels = clips / "labels.csv"
    new = not labels.exists()
    f = open(labels, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    prompts = PROMPTS
    if args.intents:
        prompts = []
        for item in args.intents.split(","):
            intent, _, state = item.strip().partition("@")
            match = [p for p in PROMPTS if p[0] == intent and (not state or p[1] == state)]
            if not match:
                sys.exit(f"unknown prompt {item!r}; options: "
                         + ", ".join(f"{p[0]}@{p[1]}" for p in PROMPTS))
            prompts.append(match[0])

    mic = Mic(device=int(args.device) if args.device and args.device.isdigit() else args.device)
    mic.start()
    print(__doc__)
    saved = 0
    for rnd in range(args.rounds):
        for intent, state, hint in prompts:
            while True:
                print(f"\n=== [{intent}] ({state}) - {hint}")
                cmd = input("  [Enter]=record  s=skip  q=quit > ").strip().lower()
                if cmd == "q":
                    print(f"saved {saved} clips"); return
                if cmd == "s":
                    break
                audio = mic.listen_ptt(prompt="  [Enter] start > ")
                dur = len(audio) / SR
                peak = float(np.max(np.abs(audio))) if len(audio) else 0
                print(f"  {dur:.1f}s, peak {peak:.2f}" + ("  (CLIPPED)" if peak > 0.99 else "")
                      + ("  (very quiet)" if peak < 0.05 else ""))
                if input("  play back? [y/N] ").strip().lower() == "y":
                    sd.play(audio, SR); sd.wait()
                k = input("  keep? [Y/n] ").strip().lower()
                if k == "n":
                    continue
                said = input("  what exactly was said (type, Enter=fill later): ").strip()
                cid = f"{args.speaker}_{args.place}_{time.strftime('%H%M%S')}_{intent}"
                path = clips / args.speaker / f"{cid}.wav"
                save_wav(path, audio)
                w.writerow({"clip_id": cid, "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                            "speaker": args.speaker, "gender": args.gender, "age": args.age,
                            "lang": args.lang, "place": args.place, "intent": intent,
                            "state": state, "transcript": said,
                            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                f.flush()
                saved += 1
                print(f"  saved {path.name}  (total this session: {saved})")
                break
    print(f"done, saved {saved} clips -> {labels}")


if __name__ == "__main__":
    main()
