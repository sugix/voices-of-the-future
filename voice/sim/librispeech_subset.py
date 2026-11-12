"""Real human speech for Chapter 4: a seeded LibriSpeech subset, clean and over a phone line.

LibriSpeech (Panayotov et al., 2015) is read English audiobook speech with human
transcripts, released under CC BY 4.0 at https://www.openslr.org/12.

Picks 40 utterances of 3 to 15 seconds from test-clean and 40 from test-other
(the harder half), then writes:

  data/librispeech/clean/<id>.wav   16 kHz, as recorded          (gitignored)
  data/librispeech/phone/<id>.wav   8 kHz mu-law phone version   (gitignored)
  data/librispeech/manifest.jsonl   ids, splits, transcripts     (committed)

Run in the nemo stack. The two parquet files (about 680 MB, from the Hugging Face copy
of LibriSpeech, which downloads far faster than openslr.org) go to a temporary folder.
Needs pyarrow.
"""

from __future__ import annotations

import io
import json
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from runtime import SCRATCH, half, quick  # noqa: E402
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from halfwheel_calls import PHONE_SR, to_phone

SEED = 2015
PER_SPLIT = 40
HF = "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/all"
CACHE = SCRATCH / "librispeech"
OUT = Path(__file__).resolve().parents[1] / "data" / "librispeech"


def fetch(split: str):
    """Yield (id, text, audio, sr) for every utterance in a LibriSpeech test split."""
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / f"{split}.parquet"
    if not local.exists():
        print(f"downloading {split}", flush=True)
        urllib.request.urlretrieve(f"{HF}/{split.replace('-', '.')}/0000.parquet", local)
    table = pq.read_table(local, columns=["id", "text", "audio"])
    for uid, text, audio in zip(*(table.column(c).to_pylist() for c in ["id", "text", "audio"])):
        x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        yield uid, text, x, sr


def main() -> None:
    rng = np.random.default_rng(SEED)
    (OUT / "clean").mkdir(parents=True, exist_ok=True)
    (OUT / "phone").mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ["test-clean", "test-other"]:
        items = [(uid, text, x, sr) for uid, text, x, sr in fetch(split) if 3.0 <= len(x) / sr <= 15.0]
        chosen = rng.choice(len(items), size=PER_SPLIT, replace=False)
        for k in sorted(chosen):
            uid, text, audio, sr = items[k]
            dur = len(audio) / sr
            sf.write(OUT / "clean" / f"{uid}.wav", audio, sr, subtype="PCM_16")
            phone = to_phone(audio, sr, float(rng.choice([15, 20, 25])), rng)
            sf.write(OUT / "phone" / f"{uid}.wav", phone, PHONE_SR, subtype="ULAW")
            rows.append({"id": uid, "split": split, "seconds": round(dur, 2), "written": text.lower()})
        print(f"{split}: {len(items)} candidates, kept {PER_SPLIT}", flush=True)
    with open(OUT / "manifest.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
