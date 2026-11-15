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
Pick **Hindi+Marathi models**, set Language to **Auto** (follows what you speak) or lock it to
one language for the whole call, and press **Start microphone**. The page shows what was heard,
which instruction was chosen and why, the current state and the beat. The **Text test** tab
runs single sentences or a whole `utterances.json` file.

## Command line
| Command | What it does |
|---|---|
| `python run.py --text in.json out.json` | Reads typed sentences, writes the chosen instructions |
| `python run.py --chat [--beat]` | Type sentences; `--beat` also plays the beat and voice |
| `python run.py --voice` | Live microphone, ignores the mic while the coach speaks |
| `python run.py --voice --ptt` | Push-to-talk (Enter to start and stop) |
| `python run.py --voice --duck 0.2` | Lowers the beat to 20% while you talk |
| `python run.py --voice --stt whisper:small@10` | Choose the speech model |
| `python run.py --text in.json out.json --nlu embed` | Use the multilingual meaning model instead |
| `python tests/test_safety.py` | Safety checks (never stop on "not", gasping or doubt) |

Text mode input and output:
```
in.json   {"items": [{"id": "T01", "state": "compressions", "text": "haath thak gaye bhai"}]}
out.json  {"results": [{"id": "T01", "intent": "tired", "reply": ["L08"], "escalate": false, "ms": 1.2}]}
```
Every item gets a result. Anything outside the script gets L11 with `escalate: true`.

## How it decides
- **Speech to text** (`coach/stt.py`). Two Whisper models, one fine-tuned on Hindi and one on
  Marathi. On the first sentence of a call, and whenever the current model is unsure (average
  log-probability below −0.5), both listen one after the other and the more confident one wins;
  the Marathi model must win by 0.15 because it is over-confident even on Hindi speech. After
  that only the winning model listens. Each model gets a short hint of the words people really
  say, which never contains the "stop" phrase.
- **Understanding** (`coach/nlu.py`). The sentence is matched against example sentences by
  character chunks, which copes with broken spellings. Keyword rules always overrule it: a "not"
  word means negation, a gasp word means gasping.
- **Choosing the instruction** (`coach/dialog.py`). States: checking, compressions, confirming
  (and stopped after L10). Every unclear or failed case ends in a keep-pushing instruction.

## Safety
L10 ("stop pushing") is said only when all of these are true:
1. the coach is in `confirming` (it has just asked "breathing normally now?");
2. the intent is `breathing_normal`;
3. no "not" word (including merged forms like "घेतने ये");
4. no doubt word ("shayad", "lagta", "maybe", "कदाचित");
5. no gasp word ("haanf", "धापा", "ajeeb", "gasping");
6. a clear positive word: हो / हाँ / yes, or a breathing verb (ले रहा, घेतोय, घेत आहे, normal);
7. speech-to-text was confident (log-probability ≥ −0.8).

Gasping is treated as not breathing. In `checking`, anything unclear starts CPR. Any error is
caught and turned into a keep-pushing instruction while the beat keeps playing.

## Recordings and measurements
- `clips/labels.csv` – one row per recording: speaker, gender, age, language, place, intent,
  state, what was said, what Whisper heard. The audio itself is not in this repository: the
  speakers agreed to recordings staying on the recording laptop.
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
| **Hindi + Marathi models** | **82 / 93** | **1** | **3.1 s** (max 10 s) | **672 MB** |

The multilingual meaning model as the line-chooser scored 39 / 93 with Whisper small (2 unsafe),
against 54 / 93 for character matching, and adds about 700 MB.

## Notes
- `script.json` is a placeholder with the expected shape (15 lines, 3 states, 13 intents).
  Replace it with the approved file; intents and IDs are read from it.
- Hindi and Marathi wording still needs review by native speakers.
