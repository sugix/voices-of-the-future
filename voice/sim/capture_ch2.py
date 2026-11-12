"""Chapter 2 capture: what common audio-handling mistakes cost a real model, plus a VAD bench.

Run on the pod from voice_applications/, in the models venv (CPU is fine):
  HF_HOME=/workspace/authorship-voice/.hf python sim/capture_ch2.py [--skip-asr]
Writes:
  data/ch2/c01_status_24k.wav  one clean call, so the chapter can show the full-band original
  data/ch2/vad_stream.wav      eight calls separated by non-speech gaps, as a phone line delivers them
  data/ch2/vad_truth.json      where the speech is, one label per 32 ms frame
  data/ch2/capture_ch2.json    Parakeet-TDT transcripts for every input mistake, and Silero VAD scores
"""

from __future__ import annotations

import argparse
import json
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from runtime import half, quick  # noqa: E402
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

CALLS = Path("data/calls")
OUT = Path("data/ch2")
SEED = 2024
SR = 8000
FRAME = 256  # 32 ms at 8 kHz: Silero VAD's frame size, and the chapter's
GAP_S = 3.0
GAPS = ["quiet line", "road noise", "hold music", "workshop clatter"] * 2


def read(path):
    x, sr = sf.read(path, dtype="float32")
    return (x.mean(axis=1) if x.ndim > 1 else x), sr


def phone_band(x):
    sos = signal.butter(4, [300, 3400], btype="bandpass", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


# ---------- Part 1: six ways to hand audio to a model ----------

def conditions(root, uid):
    """Each condition returns the float32 array the model receives, which it assumes is 16 kHz."""
    clean, csr = read(root / "clean" / f"{uid}.wav")  # 24 kHz calls, 16 kHz LibriSpeech
    phone, psr = read(root / "phone" / f"{uid}.wav")  # 8 kHz mu-law, decoded to float
    third = lambda x, sr: (x[:: sr // 8000], 8000)  # 'downsampling' by slicing
    up = lambda x, sr: signal.resample_poly(x, 16000, sr).astype(np.float32)
    return {
        "clean, resampled to 16 kHz": up(clean, csr),
        "phone, resampled to 16 kHz": up(phone, psr),
        "phone, not resampled (8 kHz read as 16 kHz)": phone.astype(np.float32),
        "clean, downsampled to 8 kHz properly": up(signal.resample_poly(clean, 8000, csr), 8000),
        "clean, downsampled to 8 kHz by slicing": up(*third(clean, csr)),
        "phone, 8x too loud and clipped": np.clip(8 * up(phone, psr), -1, 1),
        "phone, int16 values never scaled to [-1, 1]": up(phone, psr) * 32768,
    }


def run_asr(sets):
    import torch
    import nemo.collections.asr as nemo_asr
    torch.set_num_threads(4)
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").eval()
    pre = model.cfg.preprocessor
    res = {"_preprocessor": {k: pre.get(k) for k in ["sample_rate", "features", "window_size", "window_stride",
                                                     "n_fft", "normalize", "dither"]}}
    for set_name, root in sets:
        for row in quick(map(json.loads, open(root / "manifest.jsonl"))):
            for name, x in conditions(root, row["id"]).items():
                t0 = time.perf_counter()
                with torch.no_grad():
                    h = model.transcribe([x], batch_size=1, verbose=False)[0]
                res.setdefault(name, {})[f"{set_name}/{row['id']}"] = {
                    "text": h.text if hasattr(h, "text") else str(h), "s": round(time.perf_counter() - t0, 3)}
            print(set_name, row["id"], "done", flush=True)
    return res


# ---------- Part 2: a VAD bench with known answers ----------

def gap(kind, rms, rng):
    n = int(GAP_S * SR) // FRAME * FRAME  # a whole number of frames
    t = np.arange(n) / SR
    if kind == "quiet line":  # the hiss of an open line between sentences
        x = rng.normal(size=n) * rms * 10 ** (-30 / 20)
    elif kind == "road noise":  # a truck passing the caller: loud, low, rising and falling
        x = np.cumsum(rng.normal(size=n))
        x = signal.sosfilt(signal.butter(2, 60, btype="highpass", fs=SR, output="sos"), x)
        x *= np.sin(np.pi * t / GAP_S) ** 2
        x *= rms * 10 ** (-3 / 20) / np.sqrt(np.mean(x**2))
    elif kind == "hold music":  # a few sustained chords, the shop's on-hold loop
        x = np.zeros(n)
        for k, chord in enumerate([(262, 330, 392), (220, 277, 330), (196, 247, 294)]):
            seg = slice(k * n // 3, (k + 1) * n // 3)
            for f in chord:
                x[seg] += np.sin(2 * np.pi * f * t[seg]) + 0.3 * np.sin(4 * np.pi * f * t[seg])
        x *= rms * 10 ** (-6 / 20) / np.sqrt(np.mean(x**2))
    else:  # workshop clatter: a mechanic tapping a wrench on a stand, sharp decaying rings
        x = rng.normal(size=n) * rms * 10 ** (-35 / 20)
        for start in rng.uniform(0.1, GAP_S - 0.3, size=9):
            i = int(start * SR)
            m = int(0.15 * SR)
            ring = np.sin(2 * np.pi * rng.uniform(1800, 3000) * np.arange(m) / SR) * np.exp(-np.arange(m) / (0.02 * SR))
            x[i:i + m] += ring * rms * 4
    return phone_band(x)


def build_vad_stream(manifest, rng):
    pieces, labels, segments = [], [], []
    for kind, row in zip(GAPS, manifest):
        phone, _ = read(CALLS / "phone" / f"{row['id']}.wav")
        clean, _ = read(CALLS / "clean" / f"{row['id']}.wav")
        rms = float(np.sqrt(np.mean(phone**2)))
        g = gap(kind, rms, rng)
        segments.append({"kind": kind, "start_s": sum(map(len, pieces)) / SR, "seconds": len(g) / SR})
        pieces.append(g)
        labels += [0] * (len(g) // FRAME)
        # Ground truth from the clean audio: a frame is speech if it is within 30 dB of the call's loudest frame.
        n = len(phone) // FRAME
        c = np.pad(clean, (0, max(0, n * 3 * FRAME - len(clean))))[: n * 3 * FRAME].reshape(-1, 3 * FRAME)
        e = 10 * np.log10(np.mean(c**2, axis=1) + 1e-12)
        speech = e > e.max() - 30
        segments.append({"kind": "speech", "id": row["id"], "start_s": sum(map(len, pieces)) / SR,
                         "seconds": n * FRAME / SR})
        pieces.append(phone[: n * FRAME])
        labels += [1 if s else -1 for s in speech]  # -1: a pause inside a call, not scored
    x = np.concatenate(pieces).astype(np.float32)
    x = 0.9 * x / np.max(np.abs(x))
    assert len(x) // FRAME == len(labels)
    return x, labels, segments


def run_silero(path):
    import torch
    from silero_vad import load_silero_vad
    x, sr = read(path)  # read back the mu-law file, exactly what the chapter reads
    model = load_silero_vad()
    model.reset_states()
    probs = []
    with torch.no_grad():
        for i in range(len(x) // FRAME):
            probs.append(round(model(torch.from_numpy(x[i * FRAME:(i + 1) * FRAME].copy()), sr).item(), 4))
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-asr", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(l) for l in open(CALLS / "manifest.jsonl")]
    shutil.copy(CALLS / "clean" / "c01_status.wav", OUT / "c01_status_24k.wav")

    x, labels, segments = build_vad_stream(manifest[:8], np.random.default_rng(SEED))
    sf.write(OUT / "vad_stream.wav", x, SR, subtype="ULAW")
    (OUT / "vad_truth.json").write_text(json.dumps({"frame": FRAME, "sr": SR, "labels": labels,
                                                    "segments": segments}))
    cap_path = OUT / "capture_ch2.json"
    cap = json.loads(cap_path.read_text()) if cap_path.exists() else {}
    import torch
    cap["env"] = {"torch": torch.__version__, "python": platform.python_version(), "device": "cpu"}
    cap["silero"] = run_silero(OUT / "vad_stream.wav")
    print("silero done", flush=True)
    cap_path.write_text(json.dumps(cap, indent=1))
    if not args.skip_asr:
        asr = run_asr([("halfwheel", CALLS), ("librispeech", Path("data/librispeech"))])
        cap["parakeet_preprocessor"] = asr.pop("_preprocessor")
        cap["asr"] = asr
        cap_path.write_text(json.dumps(cap, indent=1))
    print("DONE capture_ch2", flush=True)


if __name__ == "__main__":
    main()
