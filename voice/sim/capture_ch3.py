"""Chapter 3 capture: transcribe the six hand-written Halfwheel calls with one model of each decoder type.

  python sim/capture_ch3.py      # needs nemo_toolkit[asr] and qwen-asr; a GPU (fp16 for a T4)

Run from voice_applications/. Reads data/calls (clean audio comes from sim/halfwheel_calls.py), writes
data/ch3/capture_ch3.json (transcripts, timings, environment) and data/ch3/parakeet_ctc_c01.npz
(CTC log-probabilities for the ticket call).
"""
import json, platform, time, traceback
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from runtime import half, quick  # noqa: E402
from pathlib import Path

import numpy as np
import torch

calls = quick([c for c in map(json.loads, open("data/calls/manifest.jsonl")) if c["id"].startswith("c0")])
out = {"env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "python": platform.python_version()}, "results": {}}


def run(name, fn):
    res = {}
    for c in calls:
        for kind in ["clean", "phone"]:
            wav = f"data/calls/{kind}/{c['id']}.wav"
            fn(wav)  # warm-up / cache on first call is excluded below
            t0 = time.perf_counter()
            text = fn(wav)
            torch.cuda.synchronize()
            res[f"{c['id']}/{kind}"] = {"text": text, "seconds": round(time.perf_counter() - t0, 3),
                                        "audio_s": c["seconds"]}
            print(f"{name:32} {c['id']:12} {kind:5} {text}", flush=True)
    out["results"][name] = res


import nemo.collections.asr as nemo_asr
out["env"]["nemo"] = nemo_asr.__version__ if hasattr(nemo_asr, "__version__") else "see pip"

for name in ["nvidia/parakeet-ctc-0.6b", "nvidia/parakeet-tdt-0.6b-v2"]:
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(name).cuda().eval()
        def tx(wav, m=model):
            h = m.transcribe([wav], verbose=False)[0]
            return h.text if hasattr(h, "text") else str(h)
        run(name, tx)
        if "ctc" in name:
            hyp = model.transcribe(["data/calls/clean/c01_status.wav"], return_hypotheses=True, verbose=False)[0]
            lp = hyp.y_sequence if hasattr(hyp, "y_sequence") else hyp.alignments
            lp = lp.detach().float().cpu().numpy() if torch.is_tensor(lp) else np.asarray(lp)
            vocab = list(model.tokenizer.vocab) if hasattr(model.tokenizer, "vocab") else \
                [model.tokenizer.ids_to_tokens([i])[0] for i in range(model.tokenizer.vocab_size)]
            np.savez_compressed("data/ch3/parakeet_ctc_c01.npz", logprobs=lp.astype(np.float16),
                                vocab=np.array(vocab + ["<blank>"]))
            print("saved CTC log-probs", lp.shape, flush=True)
        del model; torch.cuda.empty_cache()
    except Exception:
        out["results"][name] = {"error": traceback.format_exc()[-1500:]}
        print(traceback.format_exc(), flush=True)

try:
    from qwen_asr import Qwen3ASRModel
    qwen = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.float16,
                                         device_map="cuda:0", max_new_tokens=256)
    run("Qwen/Qwen3-ASR-1.7B (fp16)", lambda wav: qwen.transcribe(audio=wav, language=None)[0].text)
    out["env"]["qwen_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
except Exception:
    out["results"]["Qwen/Qwen3-ASR-1.7B (fp16)"] = {"error": traceback.format_exc()[-1500:]}
    print(traceback.format_exc(), flush=True)

Path("data/ch3/capture_ch3.json").write_text(json.dumps(out, indent=1))
print("DONE capture", flush=True)
