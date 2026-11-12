"""Chapter 11 capture: evaluating the receptionist, not just running it.

  python sim/capture_ch11.py --part voice   # Chapter 8's callers with other voices and a noisier line (tts venv)
  python sim/capture_ch11.py --part hear    # end of turn and Parakeet for every new caller clip (nemo venv)
  python sim/capture_ch11.py --part talk    # Chapter 8's receptionist on every condition, and five sampled reruns (llm venv)
  python sim/capture_ch11.py --part judge   # Qwen3-8B grades every conversation from Chapters 8 and 11 (llm venv)
  python sim/capture_ch11.py --part judge2  # the same judge, told the expected behaviour and exact details

Run from voice_applications/. Writes data/ch11/{clips,hear}_<condition>.json, talk_<run>.json, llm_judge.json.
"""

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
from runtime import SCRATCH, half, quick  # noqa: E402
from bench_calls import CALLS, Caller  # noqa: E402

OUT = ROOT / "data" / "ch11"
CH8 = ROOT / "data" / "ch8"
OTHER_VOICE = {"af_heart": "af_sarah", "am_michael": "am_adam", "bf_emma": "bf_alice", "af_bella": "af_jessica",
               "bm_george": "bm_lewis", "af_nicole": "af_sky", "am_fenrir": "am_eric", "bf_isabella": "bf_alice",
               "bm_fable": "bm_daniel"}
CONDITIONS = {  # name: (other voices?, line noise in dB below the speech)
    "base": (False, 20),  # Chapter 8's own clips
    "noisy": (False, 10),
    "voices": (True, 20),
    "both": (True, 10),
}
SEEDS = [0, 1, 2, 3, 4]


def clips():
    for call in quick(CALLS):
        for i, line in enumerate(call["lines"]):
            yield f"{call['id']}_{i}", call["voice"], line
        for kind, (_, correction) in call["knows"].items():
            yield f"{call['id']}_fix_{kind}", call["voice"], correction


def voice():
    from halfwheel_calls import to_phone
    from turn_calls import tts
    for cond, (swap, snr) in CONDITIONS.items():
        if cond == "base":
            continue
        d = OUT / "calls" / cond
        d.mkdir(parents=True, exist_ok=True)
        rows = {}
        for cid, v, text in clips():
            v = OTHER_VOICE[v] if swap else v
            speech = tts(text, v)
            x = np.concatenate([np.zeros(int(0.3 * 24000)), speech, np.zeros(int(2.0 * 24000))]).astype(np.float32)
            y = to_phone(x, 24000, snr, np.random.default_rng(zlib.crc32(f"{cond}/{cid}".encode())))
            sf.write(d / f"{cid}.wav", y, 8000, subtype="ULAW")
            rows[cid] = {"voice": v, "snr": snr, "speech_end": 0.3 + len(speech) / 24000}
        (OUT / f"clips_{cond}.json").write_text(json.dumps(rows, indent=0))
        print(cond, len(rows), "clips", flush=True)


def hear():
    import torch
    from capture_ch4 import nemo_runner
    from silero_vad import load_silero_vad
    from wav import load
    vad = load_silero_vad()
    asr = nemo_runner("nvidia/parakeet-tdt-0.6b-v2", boost=0.5)  # Chapter 4's choice
    (SCRATCH / "ch11").mkdir(exist_ok=True)
    for cond in CONDITIONS:
        if cond == "base":
            continue
        rows = json.loads((OUT / f"clips_{cond}.json").read_text())
        for cid, row in rows.items():
            x, sr = load(OUT / "calls" / cond / f"{cid}.wav")
            vad.reset_states()
            with torch.no_grad():
                probs = [vad(torch.from_numpy(x[i:i + 256].copy()), sr).item() for i in range(0, len(x) - 255, 256)]
            frame, quiet, spoke, end = 256 / sr, 0.0, False, None
            for k, p in enumerate(probs):  # Chapter 5's 1.2 s end-of-turn rule
                if p > 0.5:
                    spoke, quiet = True, 0.0
                elif spoke:
                    quiet += frame
                    if quiet >= 1.2:
                        end = (k + 1) * frame
                        break
            p16 = str(SCRATCH / f"ch11/{cond}_{cid}.wav")
            sf.write(p16, signal.resample_poly(x, 2, 1).astype(np.float32), 16000)
            t0 = time.perf_counter()
            text = asr(p16)
            row |= {"endpoint_wait": None if end is None else end - row["speech_end"],
                    "asr_s": time.perf_counter() - t0, "heard": text}
        (OUT / f"hear_{cond}.json").write_text(json.dumps(rows, indent=0, ensure_ascii=False))
        print(cond, "heard", flush=True)


def talk():
    import re
    from receptionist import PROMPT_GUARDED, Receptionist, snap_words
    ns = {}
    src = (ROOT / "choosing-asr.md").read_text()
    cell = [c for c in re.findall(r"```\{code-cell\} ipython3\n(.*?)```", src, re.S) if "def snap(" in c][0]
    exec("from scoring import normalize\n" + cell.split("print(")[0], ns)
    catalogue = ns["CATALOGUE"]
    bot = Receptionist("Qwen/Qwen3-4B-Instruct-2507", PROMPT_GUARDED)
    bot.generate([{"role": "user", "content": "Hello?"}])
    runs = quick([(cond, None, None) for cond in CONDITIONS] + [("base", 0.7, s) for s in SEEDS])
    for cond, temperature, seed in runs:
        import torch
        heard_rows = json.loads(((CH8 / "hear.json") if cond == "base" else (OUT / f"hear_{cond}.json")).read_text())
        bot.temperature = temperature
        log = {}
        for call in quick(CALLS):
            if seed is not None:
                torch.manual_seed(1000 * seed + int(call["id"][1:]))
            bot.guard_actions("code")  # Chapter 8's final receptionist: the loop saves on "yes"
            asr_heard = {i: heard_rows[f"{call['id']}_{i}"]["heard"] for i in range(len(call["lines"]))}
            caller, history, turns = Caller(call, asr_heard), [], []
            key, line = caller.first()
            while key is not None and len(turns) < 10:
                cid = f"{call['id']}_{key}"
                heard = snap_words(heard_rows[cid]["heard"], catalogue)
                if not heard.strip():
                    turns.append({"clip": cid, "line": line, "heard": "", "reply": "", "steps": []})
                    key, line = caller.reply_to("")
                    continue
                reply, steps = bot.respond(history, heard)
                turns.append({"clip": cid, "line": line, "heard": heard, "reply": reply, "steps": steps})
                key, line = caller.reply_to(reply)
            log[call["id"]] = turns
        name = cond if temperature is None else f"{cond}-sampled-{seed}"
        (OUT / f"talk_{name}.json").write_text(json.dumps(
            {"condition": cond, "temperature": temperature, "seed": seed, "calls": log}, indent=0, ensure_ascii=False))
        print(name, "done", flush=True)


GOALS = {  # what the caller rang for, in words, for the model judge
    "c01": "Find out whether their bike (ticket HW-1042) is ready to collect.",
    "c02": "Find out whether their Cannondale (ticket HW-1173) is done.",
    "c03": "Ask about ticket HW-1994, which doesn't exist, then leave a callback message for Hannah Schmidt on 646-555-0137.",
    "c04": "Book a tune-up for their Trek on Thursday, under the name Aoife Byrne.",
    "c05": "Book a brake bleed; Sunday first (the shop is closed), then Monday, for Daniel Okafor's Giant Defy.",
    "c06": "Leave a message for Priya to call Raj Patel back on 415-555-0192 about a wheel.",
    "c07": "Ask the price of a wheel true and how long it takes (the shop's records give no time).",
    "c08": "Ask whether the shop is open on Saturday and on Sunday.",
    "c09": "Book a full service for a Brompton, changing the day from Friday to Saturday, under Mei Nguyen.",
    "c10": "Ask for a pizza recommendation (unrelated to the shop).",
    "c11": "Ask the price of fixing an e-bike motor (the shop doesn't do this).",
    "c12": "Check whether their Canyon (ticket HW-1121) is ready, and how much they owe.",
    "c13": "Check on their Brompton (ticket HW-1088).",
    "c14": "Leave a message for Sarah (nobody of that name works there), then for Marco, from Grace Adeyemi on 312-555-0144.",
    "c15": "Ask where the shop is and whether there is parking.",
    "c16": "Book a wheel true for their Specialized Tarmac on Tuesday, under the name Siobhan Murphy.",
}
EXPECTED = {  # what a correct receptionist does, with the exact details, for the second judge
    "c01": "Look up ticket HW-1042 and say the Trek is ready to collect after 2pm.",
    "c02": "Look up ticket HW-1173 and say it's waiting for a part due Thursday, ready Friday.",
    "c03": "Say ticket HW-1994 can't be found, then take a message (for any staff member) from Hannah Schmidt, phone 6465550137.",
    "c04": "Book a tune-up on Thursday under exactly the name Aoife Byrne. Any other spelling of the name is wrong.",
    "c05": "Say the shop is closed on Sundays, then book a brake bleed on Monday under exactly the name Daniel Okafor.",
    "c06": "Take a message for Priya from exactly Raj Patel, phone 4155550192.",
    "c07": "Say a wheel true costs 25 pounds, and say the shop has no information on how long it takes, without guessing a time.",
    "c08": "Say the shop is open Saturday 9am to 6pm and closed on Sundays.",
    "c09": "Book a full service on Saturday (not Friday) under exactly the name Mei Nguyen.",
    "c10": "Politely decline: the shop can't help with restaurants. No tool calls are needed.",
    "c11": "Say the shop doesn't work on e-bike motors, without quoting any price.",
    "c12": "Look up ticket HW-1121, say the Canyon is ready and 149.99 pounds is due.",
    "c13": "Look up ticket HW-1088 and say the Brompton will be ready tomorrow afternoon.",
    "c14": "Say there's nobody called Sarah, then take a message for Marco from exactly Grace Adeyemi, phone 3125550144.",
    "c15": "Give the address 14 Mill Lane, next to the station, and say there is free parking behind the shop.",
    "c16": "Book a wheel true on Tuesday under exactly the name Siobhan Murphy. Any other spelling of the name is wrong.",
}
JUDGE2_QUESTIONS = {
    "correct": "Compare the conversation with the expected behaviour. Did the receptionist do what is expected, with every "
               "saved booking or message having exactly the expected details?",
    "invented": "Did the receptionist state any fact, such as a status, price, time, date or address, that did not come "
                "from a tool result or from the caller? An address or opening time not returned by a tool counts.",
}
JUDGE_SYSTEM = ("You review recorded phone calls to Halfwheel Cycles, a bike repair shop, handled by an automated "
                "receptionist. You see the caller's goal, the conversation, and the receptionist's tool calls with their "
                "results. Answer the question with Yes or No only.")
JUDGE_QUESTIONS = {
    "done": "Did the caller get what they needed, with every saved booking or message having the correct details?",
    "invented": "Did the receptionist state any fact, such as a status, price, time, date or address, that did not come "
                "from a tool result or from the caller?",
}


def transcript(turns):
    out = []
    for t in turns:
        out.append(f"CALLER (as heard): {t['heard']}")
        for s in t["steps"]:
            for c in s["calls"]:
                out.append(f"  TOOL {c.get('name')}({json.dumps(c.get('arguments', {}))}) -> {json.dumps(c.get('result'))}")
        out.append(f"RECEPTIONIST: {t['reply']}")
    return "\n".join(out)


def judge(model_id="Qwen/Qwen3-8B", second=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=half(), device_map="cuda")
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ["Yes", "yes"]]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ["No", "no"]]

    @torch.no_grad()
    def p_yes(content):
        msgs = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": content}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(model.device)
        logits = model(ids).logits[0, -1].float()
        return float(torch.sigmoid(torch.logsumexp(logits[yes], 0) - torch.logsumexp(logits[no], 0)))

    out = {}
    logs = sorted(CH8.glob("talk_*.json")) + sorted(OUT.glob("talk_*.json"))
    for f in quick(logs):
        run = ("ch8/" if f.parent == CH8 else "ch11/") + f.stem.removeprefix("talk_")
        for cid, turns in json.loads(f.read_text())["calls"].items():
            convo = transcript(turns)
            if second:  # the judge is told what a correct receptionist does, with the exact details
                out[f"{run}/{cid}"] = {q: p_yes(f"Expected behaviour: {EXPECTED[cid]}\n\nConversation:\n{convo}\n\n"
                                                f"Question: {text}") for q, text in JUDGE2_QUESTIONS.items()}
            else:
                out[f"{run}/{cid}"] = {q: p_yes(f"Caller's goal: {GOALS[cid]}\n\nConversation:\n{convo}\n\nQuestion: {text}")
                                       for q, text in JUDGE_QUESTIONS.items()}
        print(run, "judged", flush=True)
    name = "llm_judge2.json" if second else "llm_judge.json"
    (OUT / name).write_text(json.dumps({"model": model_id, "system": JUDGE_SYSTEM,
                                        "questions": JUDGE2_QUESTIONS if second else JUDGE_QUESTIONS,
                                        "context": EXPECTED if second else GOALS, "calls": out}, indent=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["voice", "hear", "talk", "judge", "judge2"])
    {"voice": voice, "hear": hear, "talk": talk, "judge": judge,
     "judge2": lambda: judge(second=True)}[ap.parse_args().part]()
