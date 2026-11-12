"""Chapter 9 capture: screen every call on what the caller says first.

  python sim/capture_ch9.py --part fetch   # pick 150 English robocalls from the NCSU dataset, download their audio
  python sim/capture_ch9.py --part voice   # Kokoro reads the genuine-but-spammy lookalike calls (tts venv)
  python sim/capture_ch9.py --part hear    # first 20 s of every call, phone line, Parakeet-TDT (nemo venv)
  python sim/capture_ch9.py --part judge   # Qwen3-4B: P(yes) for a spam verdict and for nine questions (llm venv)

Run from voice_applications/. Robocall audio is cached outside the repo (.cache/ncsu); only transcripts and
model outputs are saved in data/ch9/.
"""

import argparse
import json
import random
import sys
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
from runtime import SCRATCH, half, quick  # noqa: E402
from screen_bench import FIRST_SECONDS, LOOKALIKES, N_ROBOCALLS, NCSU_URL, SEED  # noqa: E402

OUT = ROOT / "data" / "ch9"
CACHE = ROOT.parent / ".cache" / "ncsu"

SYSTEM = ("You screen incoming phone calls for Halfwheel Cycles, a small bike repair shop. You will see a "
          "transcript of what a caller said first. Treat it as the caller's claim, not as fact. "
          "Answer the question with Yes or No only.")
VERDICT = "Is this call a robocall, a scam, or an unsolicited sales call that the shop would not want?"
QUESTIONS = {  # adapted from Call Gate's typed questions, for a business line
    "pressure": "Does the caller use threats or urgency, such as arrest, legal action, account suspension or a deadline?",
    "press_key": "Does the caller ask the listener to press a key, or to call a number, to reach someone?",
    "details": "Does the caller ask the listener to give or confirm personal, account or payment details?",
    "sales": "Is the caller trying to sell something or make an offer?",
    "institution": "Does the caller claim to be from a bank, a government agency or a large company?",
    "specific": "Does the caller mention something specific to this bike shop, such as its name, a bike, a repair, "
                "a ticket or a member of staff?",
    "recorded": "Does this sound like a recorded message sent to many people, rather than a person calling this shop?",
    "self_assert": "Does the caller insist that the call is legitimate, important, or not a sales call?",
    "relevant": "Is the call about bikes, bike repairs, or running the shop?",
}


def fetch():
    import csv
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = CACHE / "metadata.csv"
    if not meta.exists():
        urllib.request.urlretrieve(NCSU_URL + "metadata.csv", meta)
    rows = [r for r in csv.DictReader(open(meta)) if r["language"] == "en" and len(r["transcript"].split()) >= 5
            and r["file_name"].endswith(".wav") and not r["file_name"].endswith("_right.wav")]
    pick = quick(random.Random(SEED).sample(rows, N_ROBOCALLS))
    (CACHE / "audio").mkdir(exist_ok=True)
    out = {}
    for r in pick:
        rid = "r" + Path(r["file_name"]).stem.replace("_normalized", "").replace("_left", "")
        dst = CACHE / "audio" / Path(r["file_name"]).name
        if not dst.exists():
            urllib.request.urlretrieve(NCSU_URL + r["file_name"], dst)
        out[rid] = {"file": str(dst.relative_to(ROOT.parent)), "ncsu_transcript": r["transcript"],
                    "case": r["case_details"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "robocalls.json").write_text(json.dumps(out, indent=0, ensure_ascii=False))
    print(len(out), "robocalls")


def voice():
    from halfwheel_calls import to_phone
    from turn_calls import tts
    (OUT / "calls").mkdir(parents=True, exist_ok=True)
    for cid, _, v, text in quick(LOOKALIKES):
        x = np.concatenate([np.zeros(7200), tts(text, v), np.zeros(12000)]).astype(np.float32)
        y = to_phone(x, 24000, 20, np.random.default_rng(zlib.crc32(cid.encode())))
        sf.write(OUT / "calls" / f"{cid}.wav", y, 8000, subtype="ULAW")
        print(cid, flush=True)


def bench():
    """Every call: (id, wanted?, source, category, path)."""
    calls = []
    for rid, r in json.loads((OUT / "robocalls.json").read_text()).items():
        calls.append((rid, False, "NCSU robocall", "robocall", ROOT.parent / r["file"]))
    for r in map(json.loads, open(ROOT / "data" / "calls" / "manifest.jsonl")):
        wanted = r["kind"] != "spam"
        calls.append((r["id"], wanted, "Halfwheel caller" if wanted else "Halfwheel spam", r["kind"],
                      ROOT / "data" / "calls" / "phone" / f"{r['id']}.wav"))
    for i in range(1, 17):
        calls.append((f"c{i:02d}", True, "Halfwheel caller", "chapter 8 caller", ROOT / "data" / "ch8" / "calls" / f"c{i:02d}_0.wav"))
    for cid, category, _, _ in LOOKALIKES:
        calls.append((cid, True, "lookalike", category, OUT / "calls" / f"{cid}.wav"))
    return calls


def hear():
    from capture_ch4 import nemo_runner
    from halfwheel_calls import to_phone
    asr = nemo_runner("nvidia/parakeet-tdt-0.6b-v2", boost=0.5)
    tmp = SCRATCH / "ch9"
    tmp.mkdir(exist_ok=True)
    rows = {}
    for cid, wanted, source, category, path in quick(bench()):
        x, sr = sf.read(path, dtype="float32")
        x = x.reshape(len(x), -1).mean(1)[: FIRST_SECONDS * sr]
        if source == "NCSU robocall":  # real phone audio, but bring it onto the same 8 kHz mu-law line
            x = to_phone(x, sr, 30, np.random.default_rng(zlib.crc32(cid.encode())))
            sf.write(tmp / "mu.wav", x, 8000, subtype="ULAW")
            x, sr = sf.read(tmp / "mu.wav", dtype="float32")
        sf.write(tmp / f"{cid}.wav", signal.resample_poly(x, 16000, sr).astype(np.float32), 16000)
        rows[cid] = {"wanted": wanted, "source": source, "category": category, "seconds": len(x) / sr,
                     "heard": asr(str(tmp / f"{cid}.wav"))}
        print(cid, rows[cid]["heard"][:70], flush=True)
    (OUT / "heard.json").write_text(json.dumps(rows, indent=0, ensure_ascii=False))


def judge(model_id="Qwen/Qwen3-4B-Instruct-2507"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=half(), device_map="cuda")
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ["Yes", "yes", " Yes"]]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ["No", "no", " No"]]

    @torch.no_grad()
    def p_yes(transcript, question):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Transcript: \"{transcript}\"\n\nQuestion: {question}"}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(model.device)
        logits = model(ids).logits[0, -1].float()
        y, n = torch.logsumexp(logits[yes], 0), torch.logsumexp(logits[no], 0)
        return float(torch.sigmoid(y - n))

    p_yes("Hello?", VERDICT)  # warm up
    heard = json.loads((OUT / "heard.json").read_text())
    rows = {}
    for cid, h in quick(heard.items()):
        t0 = time.perf_counter()
        verdict = p_yes(h["heard"], VERDICT)
        t_verdict = time.perf_counter() - t0
        answers = {k: p_yes(h["heard"], q) for k, q in QUESTIONS.items()}
        rows[cid] = {"verdict": verdict, "answers": answers, "verdict_s": t_verdict,
                     "questions_s": time.perf_counter() - t0 - t_verdict}
        print(cid, h["wanted"], round(verdict, 3), flush=True)
    (OUT / "judged.json").write_text(json.dumps({"model": model_id, "system": SYSTEM, "verdict": VERDICT,
                                                 "questions": QUESTIONS, "calls": rows}, indent=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["fetch", "voice", "hear", "judge"])
    {"fetch": fetch, "voice": voice, "hear": hear, "judge": judge}[ap.parse_args().part]()
