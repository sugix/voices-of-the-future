"""Chapter 4 capture: every ASR model on every test file, with and without vocabulary hints.

Run on a GPU from the book's folder (it reads data/calls and data/librispeech):
  python sim/capture_ch4.py [--only name ...]
Writes data/ch4/capture_ch4.json: transcripts and per-file timings for each
model, plus peak GPU memory and the environment. Results are saved after every
model, so a crash never loses earlier work. fp16 everywhere, because the
reader's free Colab T4 has no bfloat16.
"""

from __future__ import annotations

import argparse
import json
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from runtime import half, quick  # noqa: E402
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal

OUT = Path("data/ch4/capture_ch4.json")


def load_sets():
    files = []
    for name, root in [("halfwheel", Path("data/calls")), ("librispeech", Path("data/librispeech"))]:
        for line in quick(open(root / "manifest.jsonl")):
            row = json.loads(line)
            for cond in ["clean", "phone"]:
                files.append({"set": name, "cond": cond, "id": row["id"],
                              "path": str(root / cond / f"{row['id']}.wav"), "audio_s": row["seconds"]})
    return files


def audio_at(path, sr):
    x, r = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return signal.resample_poly(x, sr, r).astype(np.float32) if r != sr else x


# Vocabulary hints: what the shop knows before the phone rings.
SHOP_WORDS = ["Halfwheel", "Trek Domane", "Specialized Tarmac", "Cannondale Synapse", "Brompton",
              "Giant Defy", "Canyon Grizl", "Surly Cross-Check", "Bianchi Oltre", "Orbea Orca",
              "Ribble", "Okafor", "Wojcik", "Nguyen", "Adeyemi", "O'Sullivan", "Mensah", "Siobhan",
              "Aoife", "Byrne", "Alvarez", "Priya"]


def nemo_runner(name, boost=None):
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict
    model = nemo_asr.models.ASRModel.from_pretrained(name).cuda().eval()
    if boost is not None:
        cfg = model.cfg.decoding
        lower = "ctc" in name  # the CTC model writes lowercase without punctuation
        phrases = [p.lower() for p in SHOP_WORDS] if lower else SHOP_WORDS
        with open_dict(cfg):
            cfg.strategy = "greedy_batch"
            # Pretrained configs have no boosting_tree section, so create it whole.
            cfg.greedy.boosting_tree = {"key_phrases_list": phrases, "context_score": 1.0, "depth_scaling": 2.0}
            cfg.greedy.boosting_tree_alpha = boost
        model.change_decoding_strategy(cfg)

    def run(path):
        h = model.transcribe([path], verbose=False)[0]
        return h.text if hasattr(h, "text") else str(h)
    return run


def qwen_runner(name):
    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained(name, dtype=torch.float16, device_map="cuda:0", max_new_tokens=256)
    return lambda path: model.transcribe(audio=path, language="English")[0].text


def whisper_runner(prompt=False):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    mid = "openai/whisper-large-v3-turbo"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(mid, torch_dtype=torch.float16).cuda().eval()
    kw = {"language": "en", "task": "transcribe"}
    if prompt:
        text = "Halfwheel Cycles bike repair. " + ", ".join(SHOP_WORDS) + "."
        kw["prompt_ids"] = torch.tensor(proc.get_prompt_ids(text)).cuda()

    def run(path):
        feats = proc(audio_at(path, 16000), sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats.cuda().half(), **kw)
        out = proc.batch_decode(ids, skip_special_tokens=True)[0]
        if prompt:  # generate() echoes the prompt before the transcript
            out = out.split(text, 1)[-1]
        return out.strip()
    return run


def kyutai_runner():
    from transformers import KyutaiSpeechToTextForConditionalGeneration, KyutaiSpeechToTextProcessor
    mid = "kyutai/stt-2.6b-en-trfs"
    proc = KyutaiSpeechToTextProcessor.from_pretrained(mid)
    model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.float16).cuda().eval()

    def run(path):
        x = audio_at(path, 24000)
        # The text stream lags the audio by 2.5 s, so append silence or the last words are lost.
        x = np.concatenate([x, np.zeros(int(3.0 * 24000), dtype=np.float32)])
        # Keep the audio float32: the Mimi codec inside the model runs in float32.
        inputs = proc(audio=x, sampling_rate=24000, return_tensors="pt").to("cuda")
        with torch.no_grad():
            ids = model.generate(**inputs)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return run


RUNNERS = {
    "parakeet-ctc-0.6b": lambda: nemo_runner("nvidia/parakeet-ctc-0.6b"),
    "parakeet-ctc-0.6b + shop words (0.5)": lambda: nemo_runner("nvidia/parakeet-ctc-0.6b", boost=0.5),
    "parakeet-ctc-0.6b + shop words (2.0)": lambda: nemo_runner("nvidia/parakeet-ctc-0.6b", boost=2.0),
    "parakeet-tdt-0.6b-v2": lambda: nemo_runner("nvidia/parakeet-tdt-0.6b-v2"),
    "parakeet-tdt-0.6b-v2 + shop words (0.5)": lambda: nemo_runner("nvidia/parakeet-tdt-0.6b-v2", boost=0.5),
    "parakeet-tdt-0.6b-v2 + shop words (2.0)": lambda: nemo_runner("nvidia/parakeet-tdt-0.6b-v2", boost=2.0),
    "whisper-large-v3-turbo": lambda: whisper_runner(),
    "whisper-large-v3-turbo + shop words": lambda: whisper_runner(prompt=True),
    "qwen3-asr-0.6b": lambda: qwen_runner("Qwen/Qwen3-ASR-0.6B"),
    "qwen3-asr-1.7b": lambda: qwen_runner("Qwen/Qwen3-ASR-1.7B"),
    "kyutai-stt-2.6b-en": kyutai_runner,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    files = load_sets()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(OUT.read_text()) if OUT.exists() else {"results": {}}
    out["results"] = {k: v for k, v in out["results"].items() if k in RUNNERS and "error" not in v}
    out["env"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                  "python": platform.python_version()}
    for name, make in RUNNERS.items():
        if args.only and name not in args.only:
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.perf_counter()
            run = make()
            load_s = time.perf_counter() - t0
            run(files[0]["path"])  # warm-up, not timed
            res = {}
            for f in files:
                t0 = time.perf_counter()
                text = run(f["path"])
                torch.cuda.synchronize()
                res[f"{f['set']}/{f['cond']}/{f['id']}"] = {"text": text, "s": round(time.perf_counter() - t0, 4),
                                                            "audio_s": f["audio_s"]}
            out["results"][name] = {"files": res, "load_s": round(load_s, 1),
                                    "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            print(f"OK {name}: {len(res)} files, peak {out['results'][name]['peak_gb']} GB", flush=True)
        except Exception:
            out["results"][name] = {"error": traceback.format_exc()[-2000:]}
            print(f"FAILED {name}\n{traceback.format_exc()}", flush=True)
        OUT.write_text(json.dumps(out, indent=1))
        run = None
        import gc; gc.collect()
    print("DONE capture_ch4", flush=True)


if __name__ == "__main__":
    main()
