"""Make the spoken lines.

  python make_voice.py --synth                 # Windows offline voices -> voices/synth/<lang>/
  python make_voice.py --record --lang hi      # record a human voice -> voices/human/hi/
  python make_voice.py --record --lang mr --only L05,L10
  python make_voice.py --female                # calm young female neural voice (hi/mr/en)
                                               #   -> voices/female/<lang>/ (needs internet ONCE)

run.py plays voices/human first, then voices/female, then voices/synth, then English, then a chime.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from coach.dialog import load_script  # noqa: E402

LANG_TAGS = {"en": ["en-in", "en-us", "en-gb", "en"], "hi": ["hi"], "mr": ["mr"]}


def _ps(script_text, timeout=120):
    import subprocess
    return subprocess.run(["powershell", "-NoProfile", "-Command", script_text],
                          capture_output=True, text=True, timeout=timeout, encoding="utf-8")


def synth(script, only):
    """Windows built-in offline voices via System.Speech (no network)."""
    import json
    import tempfile
    r = _ps("Add-Type -AssemblyName System.Speech; "
            "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | "
            "% { $_.VoiceInfo.Name + '|' + $_.VoiceInfo.Culture.Name }")
    installed = [l.split("|") for l in r.stdout.splitlines() if "|" in l]
    print("installed voices:", installed)
    for lang, tags in LANG_TAGS.items():
        v = next((name for tag in tags for name, cul in installed if cul.lower().startswith(tag)), None)
        if not v:
            print(f"{lang}: no offline voice installed - skipped (record a human voice instead,"
                  f" or add the Windows '{lang}' speech pack)")
            continue
        out = ROOT / "voices" / "synth" / lang
        out.mkdir(parents=True, exist_ok=True)
        jobs = {lid: line[lang] for lid, line in script["lines"].items()
                if (not only or lid in only) and lang in line}
        tmp = Path(tempfile.mkdtemp()) / "jobs.json"
        tmp.write_text(json.dumps({str(out / f"{k}.wav"): t for k, t in jobs.items()},
                                  ensure_ascii=False), encoding="utf-8")
        _ps("Add-Type -AssemblyName System.Speech; "
            f"$j = Get-Content -Raw -Encoding UTF8 '{tmp}' | ConvertFrom-Json; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SelectVoice('{v}'); $s.Rate = 0; "
            "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
            "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
            "[System.Speech.AudioFormat.AudioChannel]::Mono); "
            "foreach ($p in $j.PSObject.Properties) { $s.SetOutputToWaveFile($p.Name, $fmt); "
            "$s.Speak($p.Value) }; $s.SetOutputToNull(); $s.Dispose()")
        print(f"{lang}: wrote {len(jobs)} lines to {out} with {v}")


# Young (mid-20s), calm female voices, one per language. Slightly slower and a touch
# brighter than default so it sounds relaxed but youthful, not robotic.
FEMALE = {"hi": "hi-IN-SwaraNeural", "mr": "mr-IN-AarohiNeural", "en": "en-IN-NeerjaNeural"}
FEMALE_STYLE = dict(rate="-8%", pitch="+2Hz", volume="+0%")


def female(script, only, langs=("hi", "mr", "en")):
    """Microsoft neural voices via edge-tts -> 16 kHz mono WAV. Run once online; playback is offline."""
    import asyncio
    import shutil
    import subprocess
    import tempfile
    import edge_tts
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found (needed to convert mp3 -> wav)")

    async def one(text, voice, wav):
        mp3 = Path(tempfile.mkdtemp()) / "a.mp3"
        await edge_tts.Communicate(text, voice, **FEMALE_STYLE).save(str(mp3))
        wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(wav)], check=True)

    async def run():
        for lang in langs:
            voice, n = FEMALE[lang], 0
            for lid, line in script["lines"].items():
                if (only and lid not in only) or lang not in line:
                    continue
                await one(line[lang], voice, ROOT / "voices" / "female" / lang / f"{lid}.wav")
                n += 1
            print(f"{lang}: wrote {n} lines with {voice} -> voices/female/{lang}/")
    asyncio.run(run())


def record(script, lang, only):
    import sounddevice as sd
    from coach.audio import SR, Mic, save_wav
    mic = Mic()
    mic.start()
    out = ROOT / "voices" / "human" / lang
    for lid, line in script["lines"].items():
        if only and lid not in only:
            continue
        while True:
            print(f"\n{lid}: {line.get(lang) or line['en']}")
            a = mic.listen_ptt(prompt="  calm, clear, firm. [Enter] start > ")
            # trim silence at both ends
            import numpy as np
            loud = np.where(np.abs(a) > 0.02)[0]
            if len(loud):
                a = a[max(0, loud[0] - 1600): loud[-1] + 3200]
            sd.play(a, SR); sd.wait()
            if input("  keep? [Y/n] ").strip().lower() != "n":
                save_wav(out / f"{lid}.wav", a)
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--female", action="store_true", help="calm young female neural voice, hi/mr/en")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--only", default="", help="e.g. L05,L10")
    args = ap.parse_args()
    script = load_script()
    only = set(filter(None, args.only.split(",")))
    if args.synth:
        synth(script, only)
    if args.record:
        record(script, args.lang, only)
    if args.female:
        female(script, only)
    if not (args.synth or args.record or args.female):
        ap.print_help()


if __name__ == "__main__":
    main()
