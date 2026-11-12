"""Chapter 6 capture: the receptionist's script through four open TTS models, then back through ASR.

  python sim/capture_ch6.py --part voice    # fetch and transcribe the p228 reference clip (nemo stack)
  python sim/capture_ch6.py --part kokoro   # Kokoro-82M, voice bf_lily (GPU, then CPU for timing)
  python sim/capture_ch6.py --part pocket   # Kyutai Pocket TTS 100M, voice "anna" (CPU)
  python sim/capture_ch6.py --part kyutai   # Kyutai TTS 1.6B, voice vctk/p228 (GPU), plus the Mimi round trip
  python sim/capture_ch6.py --part csm      # Sesame CSM-1B, cloned from vctk/p228 (GPU)
  python sim/capture_ch6.py --part asr      # Parakeet-TDT transcribes everything, wideband and over the phone line

Each model runs in its own stack (see the chapter's note). The script and the text normaliser are read
from how-tts-works.md, so the book and this capture cannot drift apart.
Run from voice_applications/. Writes data/ch6/synth_<part>.json, capture_asr.json and data/ch6/listen/.
Generated audio goes to data/ch6/wav/ (gitignored: about 100 MB, reproducible from this script).
"""

import argparse
import json
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from runtime import SCRATCH, existing, half, merge_json, quick  # noqa: E402
OUT = Path("data/ch6")
WAV = OUT / "wav"
SEEDS = [0, 1, 2]  # sampled models read the normalised script three times
VOICE_WAV = OUT / "voice" / "p228_023_enhanced.wav"  # CSM's reference clip: VCTK speaker p228
LISTEN = ["r05"]  # the line whose audio the chapter plays, from every model


def chapter():
    src = Path("how-tts-works.md").read_text()
    ns = {}
    for cell in re.findall(r"```\{code-cell\} ipython3\n(.*?)```", src, re.S):
        if "LINES = [" in cell or "def speakable" in cell:
            exec(cell.split("\nfor lid, text in LINES")[0], ns)
    return ns["LINES"], ns["speakable"]


def jobs(sampled):
    """(line id, variant, seed, text): every line as written, then as speakable(), several times if sampled."""
    lines, speakable = chapter()
    lines = quick(lines)
    out = [(lid, "raw", 0, text) for lid, text in lines]
    for seed in SEEDS if sampled else [0]:
        out += [(lid, "spoken", seed, speakable(text)) for lid, text in lines]
    return out


def save(part, rows, meta):
    """Write this model's clips over the saved ones, keeping any saved clip that wasn't rerun."""
    path = OUT / f"synth_{part}.json"
    old = json.loads(path.read_text())["clips"] if path.exists() else {}
    path.write_text(json.dumps({"meta": meta, "clips": old | rows}, indent=0))


def write(part, lid, variant, seed, audio, sr):
    path = WAV / part / f"{lid}_{variant}_{seed}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1, 1).astype(np.float32), sr, subtype="PCM_16")
    return str(path)


def kokoro():
    import torch
    from kokoro import KPipeline
    rows, meta = {}, {"model": "hexgrad/Kokoro-82M", "voice": "bf_lily"}
    for device in ["cuda", "cpu"]:
        if device == "cpu":
            torch.set_num_threads(4)
        pipe = KPipeline(lang_code="b", device=device)
        list(pipe("Warming up.", voice="bf_lily"))
        for lid, variant, seed, text in jobs(sampled=False):
            key = f"kokoro/{lid}_{variant}_{seed}"
            t0, first, chunks, phonemes = time.perf_counter(), None, [], []
            for _, ps, audio in pipe(text, voice="bf_lily"):
                first = first or time.perf_counter() - t0
                chunks.append(np.asarray(audio, dtype=np.float32))
                phonemes.append(ps)
            wall = time.perf_counter() - t0
            a = np.concatenate(chunks)
            if device == "cuda":
                rows[key] = {"model": "kokoro", "line": lid, "variant": variant, "seed": seed, "text": text,
                             "phonemes": " ".join(phonemes), "sr": 24000, "seconds": len(a) / 24000,
                             "wav": write("kokoro", lid, variant, seed, a, 24000), "gpu": {"ttfa": first, "wall": wall}}
            else:
                rows[key]["cpu"] = {"ttfa": first, "wall": wall, "threads": 4}
            print(key, device, round(wall, 3), flush=True)
    meta["gpu_name"] = torch.cuda.get_device_name(0)
    save("kokoro", rows, meta)


def pocket():
    import torch
    from pocket_tts import TTSModel
    torch.set_num_threads(2)  # Kyutai's own claim is "uses only 2 CPU cores"
    model = TTSModel.load_model()
    state = model.get_state_for_audio_prompt("anna")  # VCTK p228, enhanced
    stream = getattr(model, "generate_audio_stream", None)
    print("streaming API:", stream is not None, flush=True)
    model.generate_audio(state, "Warming up.")
    rows = {}
    for lid, variant, seed, text in jobs(sampled=True):
        key = f"pocket/{lid}_{variant}_{seed}"
        torch.manual_seed(seed)
        t0, first = time.perf_counter(), None
        if stream:
            chunks = []
            for c in stream(state, text):
                first = first or time.perf_counter() - t0
                chunks.append(c.detach().cpu().numpy().reshape(-1))
            a = np.concatenate(chunks)
        else:
            a = model.generate_audio(state, text).detach().cpu().numpy().reshape(-1)
        wall = time.perf_counter() - t0
        sr = model.sample_rate
        rows[key] = {"model": "pocket", "line": lid, "variant": variant, "seed": seed, "text": text, "sr": sr,
                     "seconds": len(a) / sr, "wav": write("pocket", lid, variant, seed, a, sr),
                     "cpu": {"ttfa": first, "wall": wall, "threads": 2}}
        print(key, round(wall, 3), flush=True)
    save("pocket", rows, {"model": "kyutai/pocket-tts-without-voice-cloning", "voice": "anna (vctk p228)"})


def kyutai():
    import torch
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel
    info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
    tts = TTSModel.from_checkpoint_info(info, n_q=32, temp=0.6, device="cuda", dtype=half())
    voice = tts.get_voice_path("vctk/p228_023.wav")
    cond = tts.make_condition_attributes([voice], cfg_coef=2.0)

    def run(text):
        first = []
        t0 = time.perf_counter()

        def on_frame(frame):
            if not first and (frame != -1).all():
                torch.cuda.synchronize()
                first.append(time.perf_counter() - t0)

        result = tts.generate([tts.prepare_script([text], padding_between=1)], [cond], on_frame=on_frame)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        with tts.mimi.streaming(1), torch.no_grad():
            pcm = np.concatenate([tts.mimi.decode(f[:, 1:, :]).cpu().numpy()[0, 0]
                                  for f in result.frames[tts.delay_steps:]])
        return pcm, (first[0] if first else None), wall

    run("Warming up.")
    rows = {}
    for lid, variant, seed, text in jobs(sampled=True):
        key = f"kyutai/{lid}_{variant}_{seed}"
        torch.manual_seed(seed)
        a, first, wall = run(text)
        rows[key] = {"model": "kyutai", "line": lid, "variant": variant, "seed": seed, "text": text, "sr": 24000,
                     "seconds": len(a) / 24000, "wav": write("kyutai", lid, variant, seed, a, 24000),
                     "gpu": {"ttfa": first, "wall": wall}}
        print(key, round(wall, 3), flush=True)
    save("kyutai", rows, {"model": DEFAULT_DSM_TTS_REPO, "voice": "vctk/p228_023.wav",
                          "gpu_name": torch.cuda.get_device_name(0), "frame_rate": tts.mimi.frame_rate,
                          "delay_steps": tts.delay_steps})
    mimi_round_trip(tts.mimi)


def mimi_round_trip(mimi):
    """Squeeze 80 LibriSpeech recordings through Mimi at 1 to 32 codebooks, for the ASR part to score."""
    import torch
    rows = {}
    ids = [json.loads(l)["id"] for l in open("data/librispeech/manifest.jsonl")]
    ids = quick(ids)
    for n in quick([1, 2, 4, 8, 16, 32]):
        mimi.set_num_codebooks(n)
        for i in ids:
            x, sr = sf.read(f"data/librispeech/clean/{i}.wav", dtype="float32")
            x = signal.resample_poly(x, 24000, sr).astype(np.float32)
            pad = (-len(x)) % 1920
            with torch.no_grad():
                codes = mimi.encode(torch.from_numpy(np.pad(x, (0, pad)))[None, None].cuda())
                y = mimi.decode(codes).cpu().numpy()[0, 0][:len(x)]
            rows[f"mimi/{n}/{i}"] = {"codebooks": n, "id": i, "codes_shape": list(codes.shape), "sr": 24000,
                                     "wav": write(f"mimi/{n}", i, "rt", 0, y, 24000)}
        print("mimi", n, "done", flush=True)
    (OUT / "synth_mimi.json").write_text(json.dumps({"meta": {"bits_per_code": 11}, "clips": rows}, indent=0))


def csm():
    import torch
    from transformers import AutoProcessor, CsmForConditionalGeneration
    repo = "unsloth/csm-1b"  # an ungated copy of sesame/csm-1b in transformers format
    processor = AutoProcessor.from_pretrained(repo)
    model = CsmForConditionalGeneration.from_pretrained(repo, device_map="cuda", dtype=torch.float32)  # stored as fp16
    ref, sr = sf.read(VOICE_WAV, dtype="float32")
    ref = signal.resample_poly(ref, 24000, sr).astype(np.float32)
    ref_text = (OUT / "voice" / "p228_023.txt").read_text().strip()
    cap = 250  # frames at 12.5 per second: stop runaway generations at 20 s

    def run(text):
        conv = [{"role": "0", "content": [{"type": "text", "text": ref_text}, {"type": "audio", "path": ref}]},
                {"role": "0", "content": [{"type": "text", "text": text}]}]
        inputs = processor.apply_chat_template(conv, tokenize=True, return_dict=True).to(model.device)
        t0 = time.perf_counter()
        audio = model.generate(**inputs, output_audio=True, max_new_tokens=cap)
        torch.cuda.synchronize()
        return audio[0].float().cpu().numpy().reshape(-1), time.perf_counter() - t0

    run("Warming up.")
    rows = {}
    for lid, variant, seed, text in jobs(sampled=True):
        key = f"csm/{lid}_{variant}_{seed}"
        torch.manual_seed(seed)
        a, wall = run(text)
        rows[key] = {"model": "csm", "line": lid, "variant": variant, "seed": seed, "text": text, "sr": 24000,
                     "seconds": len(a) / 24000, "hit_cap": len(a) >= cap * 1920 - 1920,
                     "wav": write("csm", lid, variant, seed, a, 24000), "gpu": {"ttfa": wall, "wall": wall}}
        print(key, round(wall, 3), flush=True)
    save("csm", rows, {"model": repo, "voice": "vctk/p228_023_enhanced.wav as context", "max_frames": cap,
                       "gpu_name": torch.cuda.get_device_name(0)})


def voice():
    """Fetch the p228 reference clip and transcribe it: CSM needs the words that go with the audio."""
    import shutil
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import hf_hub_download
    VOICE_WAV.parent.mkdir(parents=True, exist_ok=True)
    VOICE_WAV.unlink(missing_ok=True)
    shutil.copyfile(hf_hub_download("kyutai/tts-voices", "vctk/p228_023_enhanced.wav"), VOICE_WAV)
    x, sr = sf.read(VOICE_WAV, dtype="float32")
    sf.write(str(SCRATCH / "p228_16k.wav"), signal.resample_poly(x.reshape(len(x), -1).mean(1), 16000, sr), 16000)
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    text = model.transcribe([str(SCRATCH / "p228_16k.wav")])[0]
    (OUT / "voice" / "p228_023.txt").write_text(getattr(text, "text", text))
    print("reference:", getattr(text, "text", text), round(len(x) / sr, 2), "s", flush=True)


def asr():
    import nemo.collections.asr as nemo_asr
    from halfwheel_calls import to_phone
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    tmp = SCRATCH / "ch6_asr"
    tmp.mkdir(exist_ok=True)
    jobs16, keys = [], []
    for f in sorted(OUT.glob("synth_*.json")):
        for key, row in existing(json.load(open(f))["clips"], "."):
            x, sr = sf.read(row["wav"], dtype="float32")
            for band in ["wide"] if key.startswith("mimi/") else ["wide", "phone"]:
                if band == "phone":  # Chapter 2's phone line at 25 dB, then G.711 mu-law
                    y = to_phone(x, sr, 25, np.random.default_rng(zlib.crc32(key.encode())))
                    sf.write(tmp / "mu.wav", y, 8000, subtype="ULAW")
                    y = signal.resample_poly(sf.read(tmp / "mu.wav", dtype="float32")[0], 2, 1)
                else:
                    y = signal.resample_poly(x, 16000, sr)
                p = tmp / f"{len(keys)}.wav"
                sf.write(p, y.astype(np.float32), 16000)
                jobs16.append(str(p))
                keys.append((key, band))
    ids = [json.loads(l)["id"] for l in open("data/librispeech/manifest.jsonl")]
    for i in ids:  # the original recordings, for the Mimi baseline
        jobs16.append(f"data/librispeech/clean/{i}.wav")
        keys.append((f"mimi/original/{i}", "wide"))
    print(len(jobs16), "files to transcribe", flush=True)
    hyps = model.transcribe(jobs16, batch_size=16)
    res = {}
    for (key, band), h in zip(keys, hyps):
        res.setdefault(key, {})[band] = getattr(h, "text", h)
    merge_json(OUT / "capture_asr.json", res)
    export_listen()


def export_listen():
    """The few clips the chapter plays, at 16 kHz to keep the repo small."""
    d = OUT / "listen"  # overwrite clip by clip, so clips from models you didn't rerun stay
    d.mkdir(exist_ok=True)
    for f in sorted(OUT.glob("synth_*.json")):
        for key, row in existing(json.load(open(f))["clips"], "."):
            m = re.fullmatch(r"(\w+)/(r\d\d)_(raw|spoken)_0", key)
            short = "mimi/" in key and row["id"] == "1320-122612-0016" and row["codebooks"] in (1, 2, 8, 32)
            if (m and m[2] in LISTEN) or short:
                x, sr = sf.read(row["wav"], dtype="float32")
                name = f"mimi_{row['codebooks']}.wav" if short else f"{m[1]}_{m[2]}_{m[3]}.wav"
                sf.write(d / name, signal.resample_poly(x, 16000, sr).astype(np.float32), 16000, subtype="PCM_16")
    runaway = WAV / "csm" / "r12_spoken_0.wav"  # a runaway: 20 s for a 5 s line
    if runaway.exists():
        x, sr = sf.read(runaway, dtype="float32")
        sf.write(d / "csm_r12_runaway.wav", signal.resample_poly(x, 8000, sr).astype(np.float32), 8000, subtype="ULAW")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["voice", "kokoro", "pocket", "kyutai", "csm", "asr", "listen"])
    part = ap.parse_args().part
    {"voice": voice, "kokoro": kokoro, "pocket": pocket, "kyutai": kyutai, "csm": csm, "asr": asr, "listen": export_listen}[part]()
