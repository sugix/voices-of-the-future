"""Chapter 7 capture: directing a voice. Five receptionist moments, read plainly and with direction.

  python sim/capture_ch7.py --part fish1    # Fish Audio S1-mini: fixed emotion tags; voice cloned from p228
  python sim/capture_ch7.py --part fish2    # Fish Audio S2-Pro: free-form tags; voice cloned from p228
  python sim/capture_ch7.py --part refs     # speaker ex04 in three moods, transcribed (nemo stack)
  python sim/capture_ch7.py --part fish2ref # S2-Pro again, with those three references
  python sim/capture_ch7.py --part kyutai   # Kyutai TTS 1.6B: style comes from the reference voice (Expresso ex04)
  python sim/capture_ch7.py --part kokoro   # Kokoro-82M: speed, and phoneme overrides for names
  python sim/capture_ch7.py --part asr      # Parakeet-TDT transcribes every clip (are the tags read out loud?)
  python sim/capture_ch7.py --part judge    # pitch, speaking rate, an emotion classifier, speaker similarity

Each part runs in its own stack (tools/stack.sh). Run from the book's folder on a GPU.
Writes data/ch7/synth_<part>.json, asr.json, judge.json and data/ch7/listen/.
Generated audio goes to data/ch7/wav/ (gitignored, reproducible from this script).
"""

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
from runtime import SCRATCH, existing, fish_paths, half, merge_json, quick  # noqa: E402
OUT = ROOT / "data" / "ch7"
WAV = OUT / "wav"
REF = ROOT / "data" / "ch6" / "voice"  # VCTK p228, the voice Chapter 6 cloned
SEEDS = [0, 1, 2]

# (scene, text, the delivery a receptionist wants, the closest class an emotion classifier knows)
SCENES = [
    ("good", "Good news! Your Brompton is ready, and it's looking brilliant.", "happy", "hap"),
    ("sorry", "I'm really sorry. We've lost the front wheel from your Canyon, and that's our fault.", "sad", "sad"),
    ("calm", "I understand, and I'm sorry it's taken so long. Let me sort it out for you right now.", "calm", "neu"),
    ("readback", "That's ticket H W one oh four two, on Thursday the fourteenth, at ten thirty.", "slow", "neu"),
    ("aside", "One moment, I'll just check that for you.", "brisk", "neu"),
]
S1_TAGS = {"good": "(joyful)", "sorry": "(sad)", "calm": "(soft tone)", "aside": "(in a hurry tone)"}
S2_TAGS = {"good": "[warm and cheerful]", "sorry": "[sincerely apologetic, quietly]",
           "calm": "[calm and reassuring]", "readback": "[slowly and clearly]", "aside": "[quickly]"}
KYUTAI_STYLE = {"good": "happy", "sorry": "sad-sympathetic", "calm": "calm", "readback": "enunciated",
                "aside": "fast"}
KOKORO_SPEED = {"readback": 0.8, "aside": 1.2}
NAMES = [  # (name, as written, with Kokoro phonemes in its British symbols, respelled for models without phonemes)
    ("Okafor", "Is that Daniel Okafor?", "Is that Daniel [Okafor](/Qkˈɑːfɔː/)?", "Is that Daniel Oh-kah-for?"),
    ("Aoife", "Is that Aoife Byrne?", "Is that [Aoife](/ˈiːfə/) Byrne?", "Is that Eefa Byrne?"),
    ("Siobhan", "Is that Siobhan Kelly?", "Is that [Siobhan](/ʃɪvˈɔːn/) Kelly?", "Is that Shiv-awn Kelly?"),
    ("Domane", "Your Trek Domane is ready.", "Your Trek [Domane](/dəmˈɑːni/) is ready.", "Your Trek Doh-mah-nee is ready."),
]


def write(part, name, audio, sr):
    path = WAV / part / f"{name}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1, 1).astype(np.float32), sr, subtype="PCM_16")
    return str(path.relative_to(ROOT))


def save(part, rows, meta):
    """Write this model's clips over the saved ones, keeping any saved clip that wasn't rerun."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"synth_{part}.json"
    old = json.loads(path.read_text())["clips"] if path.exists() else {}
    path.write_text(json.dumps({"meta": meta, "clips": old | rows}, indent=0))


def phone_ref():
    """p228 as a caller would send it: Chapter 2's phone line, then mu-law, then back up to 44.1 kHz."""
    from halfwheel_calls import to_phone
    x, sr = sf.read(REF / "p228_023_enhanced.wav", dtype="float32")
    y = to_phone(x, sr, 25, np.random.default_rng(7))
    sf.write(str(SCRATCH / "p228_phone.wav"), y, 8000, subtype="ULAW")
    y = sf.read(str(SCRATCH / "p228_phone.wav"), dtype="float32")[0]
    sf.write(str(SCRATCH / "p228_phone_44k.wav"), signal.resample_poly(y, 441, 80).astype(np.float32), 44100)
    return SCRATCH / "p228_phone_44k.wav"


def fish(part, code_dir, ckpt):
    """Both Fish models through Fish's own inference engine: a Llama-style model queue plus a DAC decoder."""
    import torch
    os.chdir(code_dir)
    sys.path.insert(0, code_dir)
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
    precision = half()
    queue = launch_thread_safe_queue(checkpoint_path=ckpt, device="cuda", precision=precision, compile=False)
    decoder = load_decoder(config_name="modded_dac_vq", checkpoint_path=f"{ckpt}/codec.pth", device="cuda")
    engine = TTSInferenceEngine(llama_queue=queue, decoder_model=decoder, precision=precision, compile=False)
    voices = {"clean": (REF / "p228_023_enhanced.wav", (REF / "p228_023.txt").read_text().strip())}
    if part == "fish2ref":
        for mood in EXPRESSO:
            voices[f"ex04_{mood}"] = (OUT / "voice" / f"ex04_{mood}.wav",
                                      (OUT / "voice" / f"ex04_{mood}.txt").read_text().strip())
    else:
        voices["phone"] = (phone_ref(), voices["clean"][1])
    tags = S1_TAGS if part == "fish1" else S2_TAGS

    def run(text, ref, seed):
        path, ref_text = voices[ref]
        req = ServeTTSRequest(text=text, references=[ServeReferenceAudio(audio=path.read_bytes(), text=ref_text)],
                              seed=seed, max_new_tokens=1024, chunk_length=200, top_p=0.7,
                              repetition_penalty=1.2, temperature=0.7, format="wav")
        t0 = time.perf_counter()
        final = [r for r in engine.inference(req) if r.code == "final"]
        wall = time.perf_counter() - t0
        if not final:
            return None, None, wall
        sr, audio = final[0].audio
        return np.asarray(audio, dtype=np.float32).reshape(-1), sr, wall

    run("Warming up.", "clean", 0)
    rows = {}
    jobs = []
    if part == "fish2ref":  # the same two lines, plain and tagged, read with references in three moods
        for scene, text, *_ in SCENES[:2]:
            for mood in EXPRESSO:
                for seed in SEEDS[:2]:
                    jobs.append((f"{scene}_plain_ex04_{mood}_{seed}", scene, "plain", f"ex04_{mood}", text, seed))
                    jobs.append((f"{scene}_directed_ex04_{mood}_{seed}", scene, "directed", f"ex04_{mood}",
                                 f"{tags[scene]} {text}", seed))
    for scene, text, *_ in ([] if part == "fish2ref" else SCENES):
        for seed in SEEDS:
            jobs.append((f"{scene}_plain_clean_{seed}", scene, "plain", "clean", text, seed))
            if scene in tags:
                jobs.append((f"{scene}_directed_clean_{seed}", scene, "directed", "clean", f"{tags[scene]} {text}", seed))
        for seed in SEEDS[:2]:
            jobs.append((f"{scene}_plain_phone_{seed}", scene, "plain", "phone", text, seed))
    for name, written, _, respelled in ([] if part == "fish2ref" else NAMES):
        jobs.append((f"name_{name}_spelled_0", f"name_{name}", "spelled", "clean", written, 0))
        jobs.append((f"name_{name}_respelled_0", f"name_{name}", "respelled", "clean", respelled, 0))
    for key, scene, variant, ref, text, seed in quick(jobs):
        audio, sr, wall = run(text, ref, seed)
        if audio is None:
            print(key, "NO AUDIO", flush=True)
            continue
        rows[f"{part}/{key}"] = {"model": part, "scene": scene, "variant": variant, "ref": ref, "seed": seed,
                                 "text": text, "sr": sr, "seconds": len(audio) / sr, "wall": wall,
                                 "wav": write(part, key, audio, sr)}
        print(key, round(len(audio) / sr, 2), "s in", round(wall, 2), flush=True)
    save(part, rows, {"checkpoint": "fishaudio/s1-mini" if part == "fish1" else "fishaudio/s2-pro", "gpu_name": torch.cuda.get_device_name(0)})


def kyutai():
    import torch
    from huggingface_hub import HfApi
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel
    tts = TTSModel.from_checkpoint_info(CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO), n_q=32, temp=0.6,
                                        device="cuda")
    files = [f for f in HfApi().list_repo_files(DEFAULT_DSM_TTS_VOICE_REPO)
             if f.startswith("expresso/") and f.endswith(".wav")]

    def voice(style):
        """One Expresso clip of speaker ex04 in the given style (ex04 is channel 1 when named first)."""
        for f in sorted(files):
            n = f.split("/")[1]
            pair, rest = n.split("_", 1)
            first, second = pair.split("-")
            if rest.startswith(style + "_") and (("channel1" in n and first == "ex04") or
                                                 ("channel2" in n and second == "ex04")):
                return f
        raise KeyError(style)

    def run(text, path):
        cond = tts.make_condition_attributes([tts.get_voice_path(path)], cfg_coef=2.0)
        result = tts.generate([tts.prepare_script([text], padding_between=1)], [cond])
        with tts.mimi.streaming(1), torch.no_grad():
            return np.concatenate([tts.mimi.decode(f[:, 1:, :]).cpu().numpy()[0, 0]
                                   for f in result.frames[tts.delay_steps:]])

    run("Warming up.", voice("default"))
    rows = {}
    for scene, text, *_ in quick(SCENES):
        for variant, style in [("plain", "default"), ("directed", KYUTAI_STYLE[scene])]:
            for seed in SEEDS:
                torch.manual_seed(seed)
                key = f"{scene}_{variant}_clean_{seed}"
                t0 = time.perf_counter()
                a = run(text, voice(style))
                rows[f"kyutai/{key}"] = {"model": "kyutai", "scene": scene, "variant": variant, "ref": "ex04",
                                         "style": style, "voice": voice(style), "seed": seed, "text": text,
                                         "sr": 24000, "seconds": len(a) / 24000, "wall": time.perf_counter() - t0,
                                         "wav": write("kyutai", key, a, 24000)}
                print(key, style, round(len(a) / 24000, 2), flush=True)
    for name, written, _, respelled in quick(NAMES):
        for variant, text in [("spelled", written), ("respelled", respelled)]:
            torch.manual_seed(0)
            key = f"name_{name}_{variant}_0"
            a = run(text, voice("default"))
            rows[f"kyutai/{key}"] = {"model": "kyutai", "scene": f"name_{name}", "variant": variant, "ref": "ex04",
                                     "seed": 0, "text": text, "sr": 24000, "seconds": len(a) / 24000,
                                     "wav": write("kyutai", key, a, 24000)}
    save("kyutai", rows, {"voice_speaker": "expresso ex04"})


def kokoro():
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="b", device="cuda")
    rows = {}

    def run(key, scene, variant, text, speed=1.0):
        chunks, phonemes = [], []
        for _, ps, audio in pipe(text, voice="bf_lily", speed=speed):
            chunks.append(np.asarray(audio, dtype=np.float32))
            phonemes.append(ps)
        a = np.concatenate(chunks)
        rows[f"kokoro/{key}"] = {"model": "kokoro", "scene": scene, "variant": variant, "seed": 0, "text": text,
                                 "speed": speed, "phonemes": " ".join(phonemes), "sr": 24000,
                                 "seconds": len(a) / 24000, "wav": write("kokoro", key, a, 24000)}

    for scene, text, *_ in quick(SCENES):
        run(f"{scene}_plain_clean_0", scene, "plain", text)
        if scene in KOKORO_SPEED:
            run(f"{scene}_directed_clean_0", scene, "directed", text, KOKORO_SPEED[scene])
        if scene == "readback":
            for speed in [0.7, 0.9, 1.1, 1.3]:
                run(f"{scene}_speed{speed}_clean_0", scene, f"speed {speed}", text, speed)
            run(f"{scene}_commas_clean_0", scene, "commas", text.replace("one oh four two", "one, oh, four, two"))
    for name, written, phonemes, _ in quick(NAMES):
        run(f"name_{name}_spelled_0", f"name_{name}", "spelled", written)
        run(f"name_{name}_phonemes_0", f"name_{name}", "phonemes", phonemes)
    save("kokoro", rows, {"voice": "bf_lily"})


EXPRESSO = {"default": "expresso/ex04-ex03_default_001_channel1_3s.wav",
            "happy": "expresso/ex04-ex02_happy_001_channel1_118s.wav",
            "sad": "expresso/ex04-ex01_sad-sympathetic_001_channel1_267s.wav"}


def refs():
    """Speaker ex04 in three moods, with transcripts, as references for Fish S2-Pro."""
    import shutil
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import hf_hub_download
    d = OUT / "voice"
    d.mkdir(parents=True, exist_ok=True)
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    for mood, f in EXPRESSO.items():
        shutil.copyfile(hf_hub_download("kyutai/tts-voices", f), d / f"ex04_{mood}.wav")
        x, sr = sf.read(d / f"ex04_{mood}.wav", dtype="float32")
        sf.write(str(SCRATCH / "ref16.wav"), signal.resample_poly(x.reshape(len(x), -1).mean(1), 16000, sr), 16000)
        text = model.transcribe([str(SCRATCH / "ref16.wav")])[0]
        (d / f"ex04_{mood}.txt").write_text(getattr(text, "text", text))
        print(mood, getattr(text, "text", text), flush=True)


def asr():
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    tmp = SCRATCH / "ch7_asr"
    tmp.mkdir(exist_ok=True)
    keys, paths = [], []
    for f in sorted(OUT.glob("synth_*.json")):
        for key, row in existing(json.load(open(f))["clips"], ROOT):
            x, sr = sf.read(ROOT / row["wav"], dtype="float32")
            p = tmp / f"{len(keys)}.wav"
            sf.write(p, signal.resample_poly(x, 16000, sr).astype(np.float32), 16000)
            keys.append(key)
            paths.append(str(p))
    hyps = model.transcribe(paths, batch_size=16)
    merge_json(OUT / "asr.json", {k: getattr(h, "text", h) for k, h in zip(keys, hyps)})


def judge():
    """Measurements a listener would make by ear: pitch, pace, loudness, emotion, and whose voice it is."""
    import librosa
    import torch
    from transformers import (AutoFeatureExtractor, AutoModelForAudioClassification, WavLMForXVector)
    ser_fe = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
    ser = AutoModelForAudioClassification.from_pretrained("superb/wav2vec2-base-superb-er").cuda().eval()
    sv_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").cuda().eval()

    def load16(path):
        x, sr = sf.read(path, dtype="float32")
        x = x.reshape(len(x), -1).mean(1)
        return signal.resample_poly(x, 16000, sr).astype(np.float32)

    @torch.no_grad()
    def embed(x):
        inputs = sv_fe(x, sampling_rate=16000, return_tensors="pt").to("cuda")
        e = sv(**inputs).embeddings[0]
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    refs = {"p228": embed(load16(REF / "p228_023_enhanced.wav"))}
    labels = ser.config.id2label
    res = {}
    for f in sorted(OUT.glob("synth_*.json")):
        for key, row in existing(json.load(open(f))["clips"], ROOT):
            x = load16(ROOT / row["wav"])
            hop = 160
            rms = librosa.feature.rms(y=x, frame_length=400, hop_length=hop)[0]
            db = 20 * np.log10(rms + 1e-6)
            voiced_frames = db > db.max() - 35
            speech_s = float(voiced_frames.sum() * hop / 16000)
            f0, vflag, _ = librosa.pyin(x, fmin=70, fmax=500, sr=16000, frame_length=1024, hop_length=hop)
            f0 = f0[vflag & ~np.isnan(f0)]
            semis = 12 * np.log2(f0 / np.median(f0)) if len(f0) else np.array([0.0])
            with torch.no_grad():
                inputs = ser_fe(x, sampling_rate=16000, return_tensors="pt").to("cuda")
                probs = torch.softmax(ser(**inputs).logits[0], -1).cpu().numpy()
            band = np.abs(np.fft.rfft(x)) ** 2
            freqs = np.fft.rfftfreq(len(x), 1 / 16000)
            res[key] = {"speech_s": speech_s, "pause_s": float(row["seconds"] - speech_s),
                        "f0_median": float(np.median(f0)) if len(f0) else None,
                        "f0_range_semitones": float(np.percentile(semis, 90) - np.percentile(semis, 10)),
                        "loudness_db": float(np.percentile(db[voiced_frames], 50)),
                        "emotion": {labels[i]: round(float(p), 4) for i, p in enumerate(probs)},
                        "above_4k": float(band[freqs > 4000].sum() / band.sum()),
                        "sim_p228": float(embed(x) @ refs["p228"])}
            print(key, "judged", flush=True)
    merge_json(OUT / "judge.json", res)


def export_listen():
    """The clips the chapter plays, at 16 kHz."""
    d = OUT / "listen"
    d.mkdir(parents=True, exist_ok=True)
    picks = ["kokoro/sorry_plain_clean_0", "fish2/sorry_plain_clean_0", "fish2/sorry_directed_clean_0",
             "fish1/sorry_directed_clean_0", "kyutai/sorry_directed_clean_0", "fish2/good_directed_clean_0",
             "kokoro/readback_plain_clean_0", "kokoro/readback_directed_clean_0", "fish2/readback_directed_clean_0",
             "fish2/good_plain_phone_0", "fish2ref/sorry_plain_ex04_sad_0", "fish2ref/sorry_plain_ex04_happy_0",
             "kokoro/name_Okafor_spelled_0", "kokoro/name_Okafor_phonemes_0"]
    rows = {}
    for f in OUT.glob("synth_*.json"):
        rows |= json.load(open(f))["clips"]
    for key in picks:
        if key in rows and (ROOT / rows[key]["wav"]).exists():
            x, sr = sf.read(ROOT / rows[key]["wav"], dtype="float32")
            sf.write(d / f"{key.replace('/', '_')}.wav", signal.resample_poly(x, 16000, sr).astype(np.float32), 16000,
                     subtype="PCM_16")
    p = SCRATCH / "p228_phone_44k.wav"
    x, sr = sf.read(p if p.exists() else phone_ref(), dtype="float32")
    sf.write(d / "ref_p228_phone.wav", signal.resample_poly(x, 16000, sr).astype(np.float32), 16000, subtype="PCM_16")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["fish1", "fish2", "fish2ref", "refs", "kyutai", "kokoro", "asr", "judge", "listen"])
    p = ap.parse_args().part
    if p == "fish1":
        fish("fish1", *fish_paths(1))
    elif p in ("fish2", "fish2ref"):
        fish(p, *fish_paths(2))
    else:
        {"refs": refs, "kyutai": kyutai, "kokoro": kokoro, "asr": asr, "judge": judge, "listen": export_listen}[p]()
