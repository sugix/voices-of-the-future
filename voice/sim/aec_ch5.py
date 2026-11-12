"""Chapter 5: echo cancellation before barge-in detection.

  python sim/aec_ch5.py --synth   # Kokoro: the receptionist's four lines, as the reference signal
  python sim/aec_ch5.py --apply   # cancel echo in every barge-in clip, then run Silero VAD on the result

--apply uses the chapter's own cancel_echo(), read from barge-in cell in listening-live.md,
so the book and this script cannot drift apart. Run in the models venv from voice_applications/.
Writes data/ch5/prompts/p<n>.wav (8 kHz) and data/ch5/capture_aec.json.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from runtime import half, quick  # noqa: E402
OUT = Path("data/ch5")


def synth():
    import turn_calls as tc
    (OUT / "prompts").mkdir(exist_ok=True)
    for i, text in enumerate(quick(tc.PROMPTS)):
        a = tc.tts(text, tc.RECEPTIONIST[0])  # same trimming as the barge-in clips
        sf.write(OUT / "prompts" / f"p{i + 1}.wav", signal.resample_poly(a, 1, 3).astype(np.float32), 8000,
                 subtype="PCM_16")
        print("prompt", i + 1, round(len(a) / 24000, 2), "s", flush=True)


def apply():
    import torch
    from silero_vad import load_silero_vad
    src = Path("listening-live.md").read_text()
    cell = next(c for c in re.findall(r"```\{code-cell\} ipython3\n(.*?)```", src, re.S) if "def cancel_echo" in c)
    from wav import load
    ns = {"np": np, "load": load}
    exec(cell.split("# --- demo")[0], ns)
    vad = load_silero_vad()
    res = {}
    for row in quick(map(json.loads, open(OUT / "manifest.jsonl"))):
        if row["set"] != "bargein":
            continue
        mic, sr = sf.read(OUT / "bargein" / f"{row['id']}.wav", dtype="float32")
        ref = ns["reference_for"](row, len(mic))
        clean = ns["cancel_echo"](mic, ref).astype(np.float32)
        vad.reset_states()
        with torch.no_grad():
            res[f"bargein/{row['id']}"] = [round(vad(torch.from_numpy(clean[i:i + 256].copy()), sr).item(), 4)
                                           for i in range(0, len(clean) - 255, 256)]
        print(row["id"], "done", flush=True)
    (OUT / "capture_aec.json").write_text(json.dumps({"frame_s": 0.032, "probs": res}, separators=(",", ":")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if a.synth:
        synth()
    if a.apply:
        apply()
