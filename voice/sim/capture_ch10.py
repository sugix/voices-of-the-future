"""Chapter 10 capture: models that take speech in and give speech out.

  python sim/capture_ch10.py --part moshi   # Moshi answers Halfwheel's callers, full duplex (kyutai venv)
  python sim/capture_ch10.py --part refs    # Expresso speaker ex01 calm and angry, transcribed (nemo venv)
  python sim/capture_ch10.py --part csm     # CSM reads one reply after a calm and after an angry caller (csm venv)

Run from voice_applications/ on a GPU pod. Writes data/ch10/moshi.json, csm.json and data/ch10/listen/.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
from runtime import SCRATCH, half, quick  # noqa: E402
OUT = ROOT / "data" / "ch10"
LEAD, TAIL, BATCH = 2.0, 8.0, 3  # silence before the caller (Moshi may greet), after, and samples per input


def inputs():
    """(id, kind, 8 kHz path, caller speech segments in seconds from the start of that file)."""
    clips = json.loads((ROOT / "data" / "ch8" / "clips.json").read_text())
    for i in range(1, 17):
        cid = f"c{i:02d}_0"
        if cid not in clips:
            continue  # Chapter 8's callers weren't all generated (a quick run)
        yield cid, "opener", ROOT / "data" / "ch8" / "calls" / f"{cid}.wav", [(0.3, clips[cid]["speech_end"])]
    for row in map(json.loads, open(ROOT / "data" / "ch5" / "manifest.jsonl")):
        if row["set"] == "turns":
            yield row["id"], "hesitant", ROOT / "data" / "ch5" / "turns" / f"{row['id']}.wav", \
                [(ph["start"], ph["end"]) for ph in row["phrases"]]


def moshi():
    import torch
    from moshi.models import LMGen, loaders
    from wav import load
    info = loaders.CheckpointInfo.from_hf_repo(os.environ.get("MOSHI_REPO", "kyutai/moshiko-pytorch-bf16"))
    mimi = info.get_mimi(device="cuda")
    tok = info.get_text_tokenizer()
    lm = info.get_moshi(device="cuda", dtype=half())
    gen = LMGen(lm, **info.lm_gen_config)
    frame = int(mimi.sample_rate / mimi.frame_rate)  # 1920 samples, 80 ms
    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    rows = {}
    for cid, kind, path, segments in quick(inputs()):
        x, sr = load(path)
        x = signal.resample_poly(x, 24000, sr).astype(np.float32)
        x = np.concatenate([np.zeros(int(LEAD * 24000)), x, np.zeros(int(TAIL * 24000))]).astype(np.float32)
        x = x[: len(x) // frame * frame]
        pcm = torch.from_numpy(x)[None, None].expand(BATCH, -1, -1).cuda()
        words = [[] for _ in range(BATCH)]
        audio = [[] for _ in range(BATCH)]
        t0 = time.perf_counter()
        with torch.no_grad(), mimi.streaming(BATCH), gen.streaming(BATCH):
            first = True
            for k, chunk in enumerate(pcm.split(frame, dim=2)):
                codes = mimi.encode(chunk)
                if first:
                    gen.step(codes)
                    first = False
                tokens = gen.step(codes)
                if tokens is None:
                    continue
                out = mimi.decode(tokens[:, 1:]).cpu().numpy()
                for b in range(BATCH):
                    t = tokens[b, 0].item()
                    if t not in (0, 3):  # 0 and 3 are padding: no new text this frame
                        words[b].append((round(k * frame / 24000, 2), tok.id_to_piece(t).replace("▁", " ")))
                    audio[b].append(out[b, 0])
        wall = time.perf_counter() - t0
        shift = [(a + LEAD, b + LEAD) for a, b in segments]
        rows[cid] = {"kind": kind, "segments": shift, "seconds": len(x) / 24000, "wall": wall,
                     "samples": [{"pieces": w, "text": "".join(p for _, p in w).strip()} for w in words]}
        y = np.concatenate(audio[0])
        sf.write(OUT / "wav" / f"moshi_{cid}.wav", np.stack([x[:len(y)], y], 1), 24000, subtype="PCM_16")
        print(cid, round(wall, 1), "s |", rows[cid]["samples"][0]["text"][:100], flush=True)
    (OUT / "moshi.json").write_text(json.dumps({"model": "kyutai/moshiko-pytorch-bf16", "lead": LEAD,
                                                "frame_s": frame / 24000, "calls": rows}, indent=0,
                                               ensure_ascii=False))


EXPRESSO = {"calm": "default", "angry": "angry"}  # speaker ex01 as the caller, in two moods


def expresso_file(style, speaker="ex01"):
    from huggingface_hub import HfApi
    for f in sorted(HfApi().list_repo_files("kyutai/tts-voices")):
        if not (f.startswith("expresso/") and f.endswith(".wav")):
            continue
        n = f.split("/")[1]
        pair, rest = n.split("_", 1)
        first, second = pair.split("-")
        if rest.startswith(style + "_") and (("channel1" in n and first == speaker) or
                                             ("channel2" in n and second == speaker)):
            return f
    raise KeyError(style)


def refs():
    import shutil
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import hf_hub_download
    d = OUT / "voice"
    d.mkdir(parents=True, exist_ok=True)
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    for mood, style in EXPRESSO.items():
        shutil.copyfile(hf_hub_download("kyutai/tts-voices", expresso_file(style)), d / f"caller_{mood}.wav")
        x, sr = sf.read(d / f"caller_{mood}.wav", dtype="float32")
        sf.write(str(SCRATCH / "ref16.wav"), signal.resample_poly(x.reshape(len(x), -1).mean(1), 16000, sr), 16000)
        text = model.transcribe([str(SCRATCH / "ref16.wav")])[0]
        (d / f"caller_{mood}.txt").write_text(getattr(text, "text", text))
        print(mood, getattr(text, "text", text), flush=True)


REPLIES = ["I'm really sorry about that. Let me check what's happened with your bike.",
           "Okay. Your bike will be ready on Friday afternoon."]


def csm():
    import librosa
    import torch
    from transformers import AutoProcessor, CsmForConditionalGeneration
    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    repo = "unsloth/csm-1b"
    processor = AutoProcessor.from_pretrained(repo)
    model = CsmForConditionalGeneration.from_pretrained(repo, device_map="cuda", dtype=torch.float32)

    def audio24(path):
        x, sr = sf.read(path, dtype="float32")
        return signal.resample_poly(x.reshape(len(x), -1).mean(1), 24000, sr).astype(np.float32)

    receptionist = (audio24(ROOT / "data" / "ch6" / "voice" / "p228_023_enhanced.wav"),
                    (ROOT / "data" / "ch6" / "voice" / "p228_023.txt").read_text().strip())
    rows = {}
    for mood in EXPRESSO:
        caller = (audio24(OUT / "voice" / f"caller_{mood}.wav")[: 24000 * 10],
                  (OUT / "voice" / f"caller_{mood}.txt").read_text().strip())
        for r, reply in enumerate(quick(REPLIES)):
            for seed in quick(range(3)):
                torch.manual_seed(seed)
                conv = [{"role": "1", "content": [{"type": "text", "text": receptionist[1]},
                                                  {"type": "audio", "path": receptionist[0]}]},
                        {"role": "0", "content": [{"type": "text", "text": caller[1]},
                                                  {"type": "audio", "path": caller[0]}]},
                        {"role": "1", "content": [{"type": "text", "text": reply}]}]
                inputs = processor.apply_chat_template(conv, tokenize=True, return_dict=True).to(model.device)
                a = model.generate(**inputs, output_audio=True, max_new_tokens=200)[0].float().cpu().numpy().reshape(-1)
                y = signal.resample_poly(a, 16000, 24000).astype(np.float32)
                f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=500, sr=16000, frame_length=1024, hop_length=160)
                f0 = f0[voiced & ~np.isnan(f0)]
                rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
                db = 20 * np.log10(rms + 1e-6)
                speech = float((db > db.max() - 35).sum() * 160 / 16000)
                key = f"{mood}_r{r}_{seed}"
                rows[key] = {"mood": mood, "reply": reply, "seed": seed, "seconds": len(a) / 24000,
                             "f0_median": float(np.median(f0)) if len(f0) else None,
                             "f0_range_semitones": float(np.ptp(12 * np.log2(np.percentile(f0, [10, 90]) / np.median(f0))))
                             if len(f0) else None,
                             "words_per_s": len(reply.split()) / speech, "loudness_db": float(np.median(db[db > db.max() - 35]))}
                sf.write(OUT / "wav" / f"csm_{key}.wav", a, 24000, subtype="PCM_16")
                print(key, {k: round(v, 2) for k, v in rows[key].items() if isinstance(v, float)}, flush=True)
    (OUT / "csm.json").write_text(json.dumps({"model": repo, "replies": REPLIES, "clips": rows}, indent=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["moshi", "refs", "csm"])
    {"moshi": moshi, "refs": refs, "csm": csm}[ap.parse_args().part]()
