# CPR Coach

An offline voice assistant that talks a bystander through hands-only CPR in Hindi, Marathi,
English and code-mixed speech. It runs on an ordinary CPU laptop with Wi-Fi off and only ever
says the 15 approved instructions in `script.json`.

```
mic ─► voice detection / push-to-talk ─► Hindi or Marathi Whisper (more confident wins)
    ─► intent (character n-gram matching + keyword safety rules) ─► state machine ─► instruction
speaker ◄── one audio stream: 110 BPM beat (never stops) + the spoken instruction
```

## Setup (once, with internet)
```
pip install -r requirements.txt
python setup_models.py        # Hindi + Marathi speech models, plain Whisper small
python make_voice.py --female # the coach's calm female voice in hi / mr / en
```
After that the coach works with Wi-Fi off.

## Use it
```
python server.py              # opens http://127.0.0.1:8765
```
The speech models load in the background as soon as the server starts (about 10 s), so Start is
instant. Pick **Hindi+Marathi models**, set Language to **Auto** (follows what you speak) or lock it to
one language for the whole call, and press **Start microphone**. The page shows what was heard,
which instruction was chosen and why, the current state and the beat. The **Text test** tab
runs single sentences or a whole `utterances.json` file; the **Recordings** tab runs audio files.

## Command line
| Command | What it does |
|---|---|
| `python run.py --text in.json out.json` | Reads typed sentences, writes the chosen instructions |
| `python run.py --chat [--beat]` | Type sentences; `--beat` also plays the beat and voice |
| `python run.py --voice` | Live microphone, ignores the mic while the coach speaks |
| `python run.py --voice --ptt` | Push-to-talk (Enter to start and stop) |
| `python run.py --voice --duck 0.2` | Lowers the beat to 20% while you talk |
| `python run.py --voice --stt whisper:small@10` | Choose the speech model |
| `python run.py --audio new_clips/ --out results.json` | Run on recordings instead of the mic |
| `python score_c.py logs/<run1> logs/<run2> logs/<run3>` | Score the speaker-feedback test (10 sentences, 3 setups) |
| `python run.py --text in.json out.json --nlu embed` | Use the multilingual meaning model instead |
| `python tests/test_safety.py` | Safety checks (never stop on "not", gasping or doubt) |

Text mode input and output:
```
in.json   {"items": [{"id": "T01", "state": "compressions", "text": "haath thak gaye bhai"}]}
out.json  {"results": [{"id": "T01", "intent": "tired", "reply": ["L08"], "escalate": false, "ms": 1.2}]}
```
Every item gets a result. Anything outside the script gets L11 with `escalate: true`.

### Other ways to test
- **Website:** run `python server.py`, open the **Text test** tab, and type one sentence or load
  an `utterances.json` file under "Batch file". You can save the results from there too.
- **Recordings:** open the **Recordings** tab and choose one or more audio files (WhatsApp
  .ogg/.opus, .m4a, .mp3, .wav). Each shows what was heard, the intent and the reply, with a
  player to listen again; save them as `results.json`. Command line:
  `python run.py --audio new_clips/ --out results.json` (`--lang mr`, `--state checking`).
- **Ready-made set:** `python run.py --text tests/utterances.json out.json` runs the 22 test
  sentences already in the project.

## How it decides
- **Speech to text** (`coach/stt.py`). Two Whisper models, one fine-tuned on Hindi and one on
  Marathi. On the first sentence of a call, and whenever the current model is unsure (average
  log-probability below −0.5), both listen one after the other and the more confident one wins;
  the Marathi model must win by 0.15 because it is over-confident even on Hindi speech. After
  that only the winning model listens. Each model gets a short hint of the words people really
  say, which never contains the "stop" phrase. Intel MKL is switched off (`CT2_USE_MKL=0`):
  with it each model reserved about 2.7 GB of memory and loading failed on an 8 GB laptop;
  without it both models need about 1.1 GB, about 20% slower.
- **Understanding** (`coach/nlu.py`). The sentence is matched against example sentences by
  character chunks, which copes with broken spellings. Keyword rules always overrule it: a "not"
  word means negation, a gasp word means gasping.
- **Choosing the instruction** (`coach/dialog.py`). States: checking, compressions, confirming
  (and stopped after L10). Every unclear or failed case ends in a keep-pushing instruction.
- **Keeping people pushing** (`coach/engine.py`). First-timers stopped pushing when the coach
  went quiet, so after 20 s of silence during CPR it says a keep-going line (L07, L13, L05).

## Safety
L10 ("stop pushing") is said only when all of these are true:
1. the coach is in `confirming` (it has just asked "breathing normally now?");
2. the intent is `breathing_normal`;
3. no "not" word (including merged forms like "घेतने ये");
4. no doubt word ("shayad", "lagta", "maybe", "कदाचित");
5. no gasp word ("haanf", "धापा", "ajeeb", "gasping");
6. a clear positive word: हो / हाँ / yes, or a breathing verb (ले रहा, घेतोय, घेत आहे, normal);
7. speech-to-text was confident (log-probability ≥ −0.8).

Gasping is treated as not breathing. "Yes, but a little" ("ho pan thoda", "haan thoda") counts as
doubt. Before CPR, a "breathing" answer is checked once more (L02), because noise can swallow
"nahi". In `checking`, anything unclear starts CPR. Any error is
caught and turned into a keep-pushing instruction while the beat keeps playing.

## Recordings and measurements
- `clips/<speaker>/*.wav` – 93 recordings from 10 speakers (ages 19–75, 6 men, 4 women;
  Marathi, Hindi and code-mixed) in a quiet room, a railway station, traffic and a market.
  Named `speaker_language_place_number_intent.wav`. All speakers agreed to sharing.
- `clips/labels.csv` – one row per recording: speaker, gender, age, language, place, intent,
  state, what was said (checked by ear), what Whisper heard.
- `clips/review.html` – open in a browser to play each clip next to its label.
- `python record_clips.py --speaker S01 --gender f --age 34 --lang mr --place traffic --consent yes`
  records new labelled clips.
- `python train.py --stt indic@10,whisper:small@10 --nlu knn,embed` measures every combination.
  Each speaker is tested by a model that never saw them. Reports go to `reports/`.
- `python train.py --stt indic@10 --apply` teaches the understanding step how the models really
  spell what people say (`data/learned_examples.json`). Transcripts that contradict their label
  are refused.

### Results on 93 real recordings (10 speakers, quiet room, station, traffic, market)
| Speech to text | Intents right | Unsafe | Time per sentence | Memory |
|---|---|---|---|---|
| Whisper tiny | 14 / 93 | 2 | 0.8 s | 111 MB |
| Whisper base | 29 / 93 | 2 | 1.7 s | 119 MB |
| Whisper small | 54 / 93 | 0 | 5.0 s (max 17 s) | 345 MB |
| Whisper small@10 | 54 / 93 | 0 | 4.7 s (max 6.9 s) | 364 MB |
| **Hindi + Marathi models** | **80 / 93** | **1** | **3.2 s** (max 12 s) | **~1.1 GB** |

Hindi + Marathi by place: quiet 28/30, railway station 13/18, traffic 30/36, market 9/9. "nahi"
missed 7 of 45 times (Whisper small: 17). The one unsafe clip is a real "shwas ghet nahi" heard as
"shwas ghetoy" at the station; the coach now asks again instead of starting CPR. With Intel MKL on
the same setup scored 82–83/93 but reserved about 5 GB; the language thresholds were tuned on
these same clips, so new voices may score lower.

The multilingual meaning model as the line-chooser scored 39 / 93 with Whisper small (2 unsafe),
against 54 / 93 for character matching, and adds about 700 MB.

## Notes
- `script.json` is a placeholder with the expected shape (15 lines, 3 states, 13 intents).
  Replace it with the approved file; intents and IDs are read from it.
- Hindi and Marathi wording still needs review by native speakers.
