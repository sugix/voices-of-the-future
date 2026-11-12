"""Chapter 5 capture: streaming recognition, end-of-turn detection and barge-in.

Run on a GPU pod from voice_applications/. Two parts, because the two toolkits want different venvs:
  python sim/capture_ch5.py --part nemo     # NeMo venv (with silero-vad): Parakeet-TDT chunked streaming,
                                            # Parakeet realtime EOU 120M, Silero VAD
  python sim/capture_ch5.py --part kyutai   # moshi venv: Kyutai STT 1B, with its end-of-turn heads
Each part writes data/ch5/capture_<part>.json, saving after every model.

Audio: Halfwheel's 30 phone calls and 80 LibriSpeech phone recordings (accuracy), plus
data/ch5/turns (callers who pause) and data/ch5/bargein (callers who interrupt), all 8 kHz mu-law.
"""

from __future__ import annotations

import argparse
import json
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from runtime import SCRATCH, half, quick  # noqa: E402
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

TMP = SCRATCH / "ch5_16k"
NEMO_STREAMING = ("https://raw.githubusercontent.com/NVIDIA/NeMo/v3.0.0/examples/asr/asr_chunked_inference/"
                  "rnnt/speech_to_text_streaming_infer_rnnt.py")
# (chunk, right context) in seconds; theoretical latency is their sum. Multiples of the 80 ms encoder frame.
CHUNKS = [(0.16, 0.16), (0.48, 0.48), (0.16, 0.8), (0.96, 0.48), (0.16, 1.28), (0.96, 0.96), (0.16, 1.84),
          (2.0, 2.0)]


def files():
    """[(key, path)] for every file this chapter uses."""
    out = []
    for name, root in [("halfwheel", Path("data/calls")), ("librispeech", Path("data/librispeech"))]:
        for row in quick(map(json.loads, open(root / "manifest.jsonl"))):
            out.append((f"{name}/{row['id']}", root / "phone" / f"{row['id']}.wav"))
    for row in quick(map(json.loads, open("data/ch5/manifest.jsonl"))):
        out.append((f"{row['set']}/{row['id']}", Path("data/ch5") / row["set"] / f"{row['id']}.wav"))
    return out


def audio_at(path, sr):
    x, r = sf.read(path, dtype="float32")
    return signal.resample_poly(x, sr, r).astype(np.float32) if r != sr else x


def save(path, data):
    path.write_text(json.dumps(data, separators=(",", ":")))


# ---------------- NeMo part ----------------

def nemo_part(out_path):
    import torch
    import nemo.collections.asr as nemo_asr
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out["env"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "python": platform.python_version()}
    TMP.mkdir(exist_ok=True)
    all_files = files()
    for key, path in all_files:  # 16 kHz copies: NeMo's scripts read files at the model's rate
        sf.write(TMP / (key.replace("/", "__") + ".wav"), audio_at(path, 16000), 16000)
    acc = [(k, TMP / (k.replace("/", "__") + ".wav")) for k, _ in all_files if k.split("/")[0] in ("halfwheel", "librispeech")]

    # 1. Parakeet-TDT offline, as the reference.
    if "tdt offline" not in out:
        model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").cuda().eval()
        t0 = time.perf_counter()
        hyps = model.transcribe([str(p) for _, p in acc], batch_size=16, verbose=False)
        out["tdt offline"] = {"latency_s": None, "wall_s": round(time.perf_counter() - t0, 1),
                              "text": {k: h.text for (k, _), h in zip(acc, hyps)}}
        save(out_path, out)
        del model
        torch.cuda.empty_cache()

    # 2. Parakeet-TDT chunked streaming, with NVIDIA's own script.
    script = SCRATCH / "nemo_streaming_rnnt.py"
    if not script.exists():
        subprocess.run(["curl", "-sL", "-o", str(script), NEMO_STREAMING], check=True)
    manifest = TMP / "acc_manifest.json"
    with open(manifest, "w") as f:
        for k, p in acc:
            f.write(json.dumps({"audio_filepath": str(p), "duration": len(sf.read(p)[0]) / 16000, "text": "x", "key": k}) + "\n")
    by_path = {str(p): k for k, p in acc}
    for chunk, right in quick(CHUNKS):
        name = f"tdt chunk {chunk}s + right {right}s"
        if name in out:
            continue
        pred = TMP / f"pred_{chunk}_{right}.json"
        t0 = time.perf_counter()
        try:
            subprocess.run([sys.executable, str(script), "pretrained_name=nvidia/parakeet-tdt-0.6b-v2",
                            f"dataset_manifest={manifest}", f"output_filename={pred}", f"chunk_secs={chunk}",
                            f"right_context_secs={right}", "left_context_secs=10.0", "batch_size=16",
                            "calculate_wer=false", "compute_dtype=float32"], check=True,
                           capture_output=True, text=True)
            rows = [json.loads(l) for l in open(pred)]
            out[name] = {"latency_s": round(chunk + right, 2), "wall_s": round(time.perf_counter() - t0, 1),
                         "text": {r.get("key") or by_path[r["audio_filepath"]]: r["pred_text"] for r in rows}}
        except subprocess.CalledProcessError as e:
            out[name] = {"error": (e.stderr or "")[-3000:]}
        print("OK" if "text" in out[name] else "FAILED", name, flush=True)
        save(out_path, out)

    # 3. Parakeet realtime EOU 120M: a cache-aware streaming model, stepped chunk by chunk.
    if "eou120m" not in out:
        try:
            out["eou120m"] = stream_eou([(k, TMP / (k.replace("/", "__") + ".wav")) for k, _ in all_files])
        except Exception:
            out["eou120m"] = {"error": traceback.format_exc()[-3000:]}
        print("eou120m", "error" not in out["eou120m"], flush=True)
        save(out_path, out)

    # 4. Silero VAD on the turn and barge-in clips, 32 ms frames at 8 kHz.
    if "silero" not in out:
        from silero_vad import load_silero_vad
        vad = load_silero_vad()
        res = {}
        for key, path in all_files:
            if key.split("/")[0] not in ("turns", "bargein"):
                continue
            x, sr = sf.read(path, dtype="float32")
            vad.reset_states()
            with torch.no_grad():
                res[key] = [round(vad(torch.from_numpy(x[i:i + 256].copy()), sr).item(), 4)
                            for i in range(0, len(x) - 255, 256)]
        out["silero"] = {"frame_s": 0.032, "probs": res}
        save(out_path, out)
    print("DONE nemo", flush=True)


def stream_eou(items):
    import torch
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    from omegaconf import open_dict
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet_realtime_eou_120m-v1").cuda().eval()
    cfg = model.cfg.decoding  # as NVIDIA's streaming script does: plain greedy, no alignments or timestamps
    with open_dict(cfg):
        cfg.strategy = "greedy_batch"
        cfg.preserve_alignments = False
        cfg.compute_timestamps = False
        cfg.fused_batch_size = -1
        if "greedy" in cfg:
            cfg.greedy.preserve_alignments = False
    model.change_decoding_strategy(cfg)
    stride = model.cfg.preprocessor.window_stride
    res = {"streaming_cfg": str(model.encoder.streaming_cfg), "files": {}}
    for key, path in items:
        buf = CacheAwareStreamingAudioBuffer(model=model, online_normalization=False, pad_and_drop_preencoded=False)
        buf.append_audio_file(str(path), stream_id=-1)
        c_ch, c_t, c_len = model.encoder.get_initial_cache_state(batch_size=1)
        prev_hyp, prev_out, events, last, step_ms = None, None, [], "", []
        for step, (chunk, lengths) in enumerate(iter(buf)):
            t0 = time.perf_counter()
            with torch.inference_mode():
                prev_out, texts, c_ch, c_t, c_len, prev_hyp = model.conformer_stream_step(
                    processed_signal=chunk, processed_signal_length=lengths, cache_last_channel=c_ch,
                    cache_last_time=c_t, cache_last_channel_len=c_len, keep_all_outputs=buf.is_buffer_empty(),
                    previous_hypotheses=prev_hyp, previous_pred_out=prev_out,
                    drop_extra_pre_encoded=0 if step == 0 else model.encoder.streaming_cfg.drop_extra_pre_encoded,
                    return_transcription=True)
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - t0) * 1000)
            text = texts[0].text if hasattr(texts[0], "text") else str(texts[0])
            heard_s = round(getattr(buf, "buffer_idx", 0) * stride, 3)  # audio consumed so far
            if text != last:
                events.append([heard_s, text])
                last = text
        res["files"][key] = {"final": last, "events": events, "steps": step + 1,
                             "median_step_ms": round(float(np.median(step_ms)), 2)}
    return res


# ---------------- Kyutai part ----------------

def kyutai_part(out_path):
    import torch
    from moshi.models import LMGen, loaders
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out["env"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "python": platform.python_version()}
    info = loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
    mimi = info.get_mimi(device="cuda")
    tok = info.get_text_tokenizer()
    lm = info.get_moshi(device="cuda", dtype=half())
    cfg = info.stt_config
    out["stt_config"] = cfg
    if len(getattr(lm, "extra_heads", [])) == 0:
        # The PyTorch release ships without the end-of-turn heads; Kyutai's Rust (candle) release has them.
        # The two checkpoints are otherwise identical, tensor for tensor, so attach the four heads.
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        w = load_file(hf_hub_download("kyutai/stt-1b-en_fr-candle", "model.safetensors", cache_dir=str(SCRATCH / "hf")))
        dtype = next(lm.parameters()).dtype
        heads = []
        for i in range(4):
            lin = torch.nn.Linear(w[f"extra_heads.{i}.weight"].shape[1], 6, bias=False)
            lin.weight.data.copy_(w[f"extra_heads.{i}.weight"])
            heads.append(lin)
        lm.extra_heads = torch.nn.ModuleList(heads).to("cuda", dtype)
        del w
    out["extra_heads"] = len(lm.extra_heads)
    frame = int(mimi.sample_rate / mimi.frame_rate)
    step_s = 1 / mimi.frame_rate
    pad_left = int(cfg.get("audio_silence_prefix_seconds", 0.0) * mimi.sample_rate)
    pad_right = int((cfg.get("audio_delay_seconds", 0.0) + 1.0) * mimi.sample_rate)
    res = {}
    for key, path in files():
        x = torch.from_numpy(audio_at(path, int(mimi.sample_rate))).cuda()
        x = torch.nn.functional.pad(x, (pad_left, pad_right))[None, None]
        gen = LMGen(lm, temp=0, temp_text=0, use_sampling=False)
        words, prs, step_ms = [], [], []
        with mimi.streaming(1), gen.streaming(1):
            first = True
            for i, c in enumerate(x.split(frame, dim=2)):
                if c.shape[-1] != frame:
                    break
                t0 = time.perf_counter()
                codes = mimi.encode(c)
                if first:  # the first step is consumed by the model's initial tokens, as in Kyutai's notebook
                    gen.step_with_extra_heads(codes)
                    first = False
                o = gen.step_with_extra_heads(codes)
                torch.cuda.synchronize()
                step_ms.append((time.perf_counter() - t0) * 1000)
                if o is None:
                    continue
                tokens, heads = o
                t = round((i + 1) * step_s - pad_left / mimi.sample_rate, 3)  # seconds of real audio heard so far
                tid = tokens[0, 0].reshape(-1)[0].item()
                if tid not in (0, 3):
                    words.append([t, tok.id_to_piece(tid).replace("▁", " ")])
                if heads:
                    prs.append([t] + [round(h.reshape(-1, h.shape[-1])[0, 0].item(), 4) for h in heads])
        res[key] = {"text": "".join(w for _, w in words).strip(), "median_step_ms": round(float(np.median(step_ms)), 2)}
        if key.split("/")[0] in ("turns", "bargein"):  # timing detail only where the chapter uses it
            res[key].update(pieces=words, prs=prs)
    out["stt-1b-en_fr"] = res
    save(out_path, out)
    print("DONE kyutai", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["nemo", "kyutai"], required=True)
    a = ap.parse_args()
    p = Path(f"data/ch5/capture_{a.part}.json")
    (nemo_part if a.part == "nemo" else kyutai_part)(p)
