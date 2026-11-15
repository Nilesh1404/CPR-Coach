"""Download the speech models once (needs internet). After this the coach runs fully offline.

  python setup_models.py

  models/whisper-hindi-small-ct2    Whisper small fine-tuned on Hindi (vasista22, CTranslate2)
  models/whisper-marathi-small-ct2  Whisper small fine-tuned on Marathi (steja), converted to int8
  Hugging Face cache                plain Whisper small (English, and for comparisons)
"""
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"


def hindi():
    out = MODELS / "whisper-hindi-small-ct2"
    if (out / "model.bin").exists() and (out / "tokenizer.json").exists():
        print("hindi: already there")
        return
    from huggingface_hub import snapshot_download
    from transformers import WhisperTokenizerFast
    snapshot_download("songzewu/vasista22-whisper-hindi-small-ct2", local_dir=out,
                      ignore_patterns=["pytorch_model.bin", "flax_model.msgpack", "training_args.bin"])
    WhisperTokenizerFast.from_pretrained(out).save_pretrained(out)
    shutil.rmtree(out / ".cache", ignore_errors=True)
    print("hindi: ready")


def marathi():
    out = MODELS / "whisper-marathi-small-ct2"
    if (out / "model.bin").exists() and (out / "tokenizer.json").exists():
        print("marathi: already there")
        return
    from huggingface_hub import snapshot_download
    from transformers import WhisperFeatureExtractor, WhisperTokenizerFast
    src = Path(snapshot_download("steja/whisper-small-marathi", local_dir=MODELS / "_src_marathi",
                                 allow_patterns=["*.json", "*.txt", "pytorch_model.bin"]))
    subprocess.run(["ct2-transformers-converter", "--model", str(src), "--output_dir", str(out),
                    "--quantization", "int8", "--force"], check=True)
    WhisperTokenizerFast.from_pretrained(src).save_pretrained(out)
    WhisperFeatureExtractor.from_pretrained(src).save_pretrained(out)
    shutil.rmtree(src, ignore_errors=True)
    print("marathi: ready")


def whisper_small():
    from faster_whisper import WhisperModel
    WhisperModel("small", device="cpu", compute_type="int8")
    print("whisper small: ready")


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True)
    hindi()
    marathi()
    whisper_small()
