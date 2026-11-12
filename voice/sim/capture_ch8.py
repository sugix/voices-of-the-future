"""Chapter 8 capture: the whole receptionist, run over sixteen calls.

  python sim/capture_ch8.py --part voice   # Kokoro reads every caller line; Chapter 2's phone line (tts venv)
  python sim/capture_ch8.py --part hear    # Silero end of turn (Ch 5) + Parakeet-TDT, boosted (Ch 4) (nemo venv)
  python sim/capture_ch8.py --part talk    # Qwen3 receptionists answer every call, with tools (llm venv)
  python sim/capture_ch8.py --part speak   # Pocket TTS and Kokoro: time to the first audio of each reply (tts venv)

Run from voice_applications/. Writes data/ch8/calls/*.wav (8 kHz mu-law), hear.json, talk_<config>.json, speak.json.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
from bench_calls import CALLS, Caller  # noqa: E402

OUT = ROOT / "data" / "ch8"
LEAD, TAIL = 0.3, 2.0  # silence before and after each caller line, in seconds

CONFIGS = {  # name: (model, how the receptionist hears the caller, instructions); grouped by model
    "4b-snap-guarded": ("Qwen/Qwen3-4B-Instruct-2507", "snap", "guarded"),
    "4b-text-guarded": ("Qwen/Qwen3-4B-Instruct-2507", "text", "guarded"),
    "4b-asr-guarded": ("Qwen/Qwen3-4B-Instruct-2507", "asr", "guarded"),
    "4b-snap-plain": ("Qwen/Qwen3-4B-Instruct-2507", "snap", "plain"),
    "4b-snap-twophase": ("Qwen/Qwen3-4B-Instruct-2507", "snap", "twophase"),
    "4b-text-twophase": ("Qwen/Qwen3-4B-Instruct-2507", "text", "twophase"),
    "4b-snap-code": ("Qwen/Qwen3-4B-Instruct-2507", "snap", "code"),
    "1.7b-snap-guarded": ("Qwen/Qwen3-1.7B", "snap", "guarded"),
    "8b-snap-guarded": ("Qwen/Qwen3-8B", "snap", "guarded"),
    "8b-snap-twophase": ("Qwen/Qwen3-8B", "snap", "twophase"),
    "8b-snap-code": ("Qwen/Qwen3-8B", "snap", "code"),
}


def clips():
    """Every caller line and correction line: (clip id, voice, text)."""
    for call in CALLS:
        for i, line in enumerate(call["lines"]):
            yield f"{call['id']}_{i}", call["voice"], line
        for kind, (_, correction) in call["knows"].items():
            yield f"{call['id']}_fix_{kind}", call["voice"], correction


def chapter_cells(md, marker):
    src = (ROOT / md).read_text()
    return [c for c in re.findall(r"```\{code-cell\} ipython3\n(.*?)```", src, re.S) if marker in c]


def voice():
    from halfwheel_calls import to_phone
    from turn_calls import tts
    (OUT / "calls").mkdir(parents=True, exist_ok=True)
    rows = {}
    for cid, v, text in clips():
        speech = tts(text, v)  # Chapter 5's Kokoro helper, trimmed to the speech
        x = np.concatenate([np.zeros(int(LEAD * 24000)), speech, np.zeros(int(TAIL * 24000))]).astype(np.float32)
        y = to_phone(x, 24000, 20, np.random.default_rng(zlib.crc32(cid.encode())))
        sf.write(OUT / "calls" / f"{cid}.wav", y, 8000, subtype="ULAW")
        rows[cid] = {"voice": v, "text": text, "speech_end": LEAD + len(speech) / 24000, "seconds": len(y) / 8000}
        print(cid, round(len(speech) / 24000, 2), flush=True)
    (OUT / "clips.json").write_text(json.dumps(rows, indent=0))


def hear():
    import torch
    from capture_ch4 import nemo_runner
    from silero_vad import load_silero_vad
    from wav import load
    vad = load_silero_vad()
    asr = nemo_runner("nvidia/parakeet-tdt-0.6b-v2", boost=0.5)  # Chapter 4's choice for live calls
    rows = json.loads((OUT / "clips.json").read_text())
    Path("/root/ch8").mkdir(exist_ok=True)
    asr(str(OUT / "calls" / f"{next(iter(rows))}.wav"))  # warm up
    for cid, row in rows.items():
        x, sr = load(OUT / "calls" / f"{cid}.wav")
        vad.reset_states()
        with torch.no_grad():
            probs = [vad(torch.from_numpy(x[i:i + 256].copy()), sr).item() for i in range(0, len(x) - 255, 256)]
        # Chapter 5's rule: the turn ends after 1.2 s without speech, once the caller has spoken
        frame, quiet, spoke, end = 256 / sr, 0.0, False, None
        for k, p in enumerate(probs):
            if p > 0.5:
                spoke, quiet = True, 0.0
            elif spoke:
                quiet += frame
                if quiet >= 1.2:
                    end = (k + 1) * frame
                    break
        p16 = f"/root/ch8/{cid}.wav"
        sf.write(p16, signal.resample_poly(x, 2, 1).astype(np.float32), 16000)
        t0 = time.perf_counter()
        text = asr(p16)
        row |= {"endpoint": end, "endpoint_wait": None if end is None else end - row["speech_end"],
                "asr_s": time.perf_counter() - t0, "heard": text}
        print(cid, round(row["endpoint_wait"] or -1, 2), text, flush=True)
    (OUT / "hear.json").write_text(json.dumps(rows, indent=0))


def talk(only=None):
    import gc
    import torch
    from receptionist import PROMPT_GUARDED, PROMPT_PLAIN, Receptionist
    from receptionist import snap_words
    ns = {}
    exec("from scoring import normalize\n" + chapter_cells("choosing-asr.md", "def snap(")[0], ns)
    catalogue = ns["CATALOGUE"]  # Chapter 4: what the shop knows before the phone rings
    hear_rows = json.loads((OUT / "hear.json").read_text())
    loaded, bot = (None, None), None
    for name, (model_id, mode, prompt) in CONFIGS.items():
        if only and name not in only:
            continue
        if loaded is None or loaded[0] != model_id:
            loaded = bot = None  # free the previous model before loading the next
            gc.collect()
            torch.cuda.empty_cache()
            loaded = (model_id, Receptionist(model_id, PROMPT_GUARDED))
            loaded[1].generate([{"role": "user", "content": "Hello?"}])  # warm up
        bot = loaded[1]
        bot.prompt = PROMPT_PLAIN if prompt == "plain" else PROMPT_GUARDED
        guard = {"twophase": "tool", "code": "code"}.get(prompt)  # the read-back rule enforced in code
        log = {}
        for call in CALLS:
            bot.guard_actions(guard)  # fresh pending state for every call
            asr_heard = {i: hear_rows[f"{call['id']}_{i}"]["heard"] for i in range(len(call["lines"]))}
            caller, history, turns = Caller(call, asr_heard), [], []
            key, line = caller.first()
            while key is not None and len(turns) < 10:
                cid = f"{call['id']}_{key}"
                heard = {"text": line, "asr": hear_rows[cid]["heard"],
                         "snap": snap_words(hear_rows[cid]["heard"], catalogue)}[mode]
                if not heard.strip():  # the recogniser heard nothing: don't ask the model to answer silence
                    turns.append({"clip": cid, "line": line, "heard": "", "reply": "", "steps": []})
                    key, line = caller.reply_to("")
                    continue
                reply, steps = bot.respond(history, heard)
                turns.append({"clip": cid, "line": line, "heard": heard, "reply": reply, "steps": steps})
                key, line = caller.reply_to(reply)
            log[call["id"]] = turns
            print(name, call["id"], len(turns), "turns |", turns[-1]["reply"][:80], flush=True)
        (OUT / f"talk_{name}.json").write_text(json.dumps(
            {"model": model_id, "mode": mode, "prompt": prompt, "gpu": torch.cuda.get_device_name(0), "calls": log},
            indent=0, ensure_ascii=False))


def speak():
    """Time to first audio for the first sentence of every reply the main receptionist spoke."""
    import torch
    from kokoro import KPipeline
    from pocket_tts import TTSModel
    ns = {}
    for cell in chapter_cells("how-tts-works.md", "def speakable"):
        exec(cell.split("\nfor lid, text in LINES")[0], ns)
    speakable = ns["speakable"]  # Chapter 6's normaliser
    talk_log = json.loads((OUT / "talk_4b-snap-guarded.json").read_text())["calls"]
    firsts = sorted({re.split(r"(?<=[.!?])\s", t["reply"].strip())[0] for turns in talk_log.values() for t in turns
                     if t["reply"].strip()})
    torch.set_num_threads(2)
    pocket = TTSModel.load_model()
    state = pocket.get_state_for_audio_prompt("anna")
    kokoro = KPipeline(lang_code="b", device="cuda")
    pocket.generate_audio(state, "Warming up.")
    list(kokoro("Warming up.", voice="bf_lily"))
    rows = {}
    for s in firsts:
        text = speakable(s)
        t0 = time.perf_counter()
        next(iter(pocket.generate_audio_stream(state, text)))
        p = time.perf_counter() - t0
        t0 = time.perf_counter()
        next(iter(kokoro(text, voice="bf_lily")))
        k = time.perf_counter() - t0
        rows[s] = {"speakable": text, "pocket_cpu": p, "kokoro_gpu": k}
    filler = "One moment, let me check."
    rows["_filler"] = {"text": filler, "note": "pre-rendered once, so it plays immediately"}
    (OUT / "speak.json").write_text(json.dumps(rows, indent=0, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["voice", "hear", "talk", "speak"])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    {"voice": voice, "hear": hear, "talk": lambda: talk(a.only), "speak": speak}[a.part]()
