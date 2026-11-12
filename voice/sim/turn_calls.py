"""Chapter 5 test audio: callers who pause mid-turn, and callers who talk over the receptionist.

Everything is synthesised with Kokoro-82M phrase by phrase, so the ground truth is exact:
we know where every phrase starts and ends, which pauses are mid-turn, and when the turn ends.

  data/ch5/turns/<id>.wav    one caller turn with hesitation pauses, then 3 s of open line
  data/ch5/bargein/<id>.wav  the receptionist speaking (as residual echo) while the caller
                             barges in, backchannels ("mm-hm"), or stays silent
  data/ch5/manifest.jsonl    timings and labels for both sets

All files are 8 kHz mu-law, through the same phone line as Chapter 2's calls.
Run on the pod in the models venv, from voice_applications/:
  python sim/turn_calls.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from runtime import half, quick  # noqa: E402
from halfwheel_calls import PHONE_SR, TTS_SR  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "ch5"
SEED = 5050
RECEPTIONIST = ("bf_lily", "b")  # a voice no caller uses

# (id, voice, phrases, pause after each phrase but the last, kind of each pause)
# "incomplete": the sentence is unfinished, so ending the turn here cuts the caller off.
# "complete": the sentence is finished but the caller has more to say. Genuinely ambiguous.
TURNS = [
    ("t01", "af_heart", ["Hi, my ticket number is", "H W one oh four two."], [0.9], ["incomplete"]),
    ("t02", "am_michael", ["I'd like to book a tune-up for my", "um, Trek Domane,", "sometime on Thursday."],
     [0.7, 1.1], ["incomplete", "incomplete"]),
    ("t03", "bf_emma", ["It's Daniel Okafor.", "I'm calling about my Brompton."], [0.8], ["complete"]),
    ("t04", "bm_george", ["Can you ask Priya to call me back on", "four one five,", "five five five,",
                          "oh one nine two."], [0.6, 0.5, 0.6], ["incomplete"] * 3),
    ("t05", "af_bella", ["So the rear wheel is making a clicking noise, and", "I think it might be the hub."],
     [1.2], ["incomplete"]),
    ("t06", "af_nicole", ["Yes, that's right.", "And could you also check the brakes?"], [1.0], ["complete"]),
    ("t07", "am_fenrir", ["My name is", "Aoife Byrne.", "A O I F E."], [0.8, 0.7], ["incomplete", "complete"]),
    ("t08", "bf_isabella", ["I need to move my booking from Saturday to", "hmm, let me check,", "Monday."],
     [1.0, 1.4], ["incomplete", "incomplete"]),
    ("t09", "bm_fable", ["Is the shop open", "on Sunday?"], [0.5], ["incomplete"]),
    ("t10", "af_heart", ["The bike is a", "Canyon Grizl."], [1.0], ["incomplete"]),
    ("t11", "am_michael", ["How much is a wheel true?"], [], []),
    ("t12", "bf_emma", ["I dropped off my Cannondale last week.", "The ticket is H W one one seven three."],
     [0.9], ["complete"]),
    ("t13", "bm_george", ["Could I pick it up", "after work, around six?"], [0.6], ["incomplete"]),
    ("t14", "af_bella", ["Thanks, that's all."], [], []),
    ("t15", "af_nicole", ["The brakes are squeaking, and the chain keeps", "slipping when I change gear."],
     [1.3], ["incomplete"]),
    ("t16", "am_fenrir", ["Okay.", "So what time should I come in?"], [0.9], ["complete"]),
]

PROMPTS = [
    "Thanks for calling Halfwheel Cycles. Your Trek Domane is booked in for a tune-up on Thursday at ten, "
    "and it should be ready by the afternoon.",
    "I can see ticket H W one oh four two. The mechanic has replaced the chain and cassette, "
    "and the bike is ready to collect any time after two.",
    "A wheel true is usually twenty five pounds, and we can normally do it while you wait, "
    "if you bring it in before midday.",
    "I'll pass that on to Priya. Can I take a number she can call you back on, and a good time to reach you?",
]
EVENTS = [  # (kind, what the caller says over the receptionist, voice)
    ("barge-in", "Sorry, it's actually Friday.", "af_heart"),
    ("barge-in", "Wait, wrong bike.", "bm_george"),
    ("barge-in", "No, the ticket is H W one oh four three.", "af_bella"),
    ("barge-in", "Can I stop you there?", "am_michael"),
    ("backchannel", "Mm-hm.", "bf_emma"),
    ("backchannel", "Okay.", "am_fenrir"),
    ("backchannel", "Yeah.", "af_nicole"),
    ("backchannel", "Right.", "bm_fable"),
    ("none", "", ""),
    ("none", "", ""),
    ("none", "", ""),
    ("none", "", ""),
]

_pipes = {}


def tts(text, voice):
    from kokoro import KPipeline
    lang = voice[0]
    pipe = _pipes.get(lang) or _pipes.setdefault(lang, KPipeline(lang_code=lang))
    audio = np.concatenate([np.asarray(a, dtype=np.float32) for _, _, a in pipe(text, voice=voice)])
    return trim(audio)


def trim(x, floor_db=-45):
    """Cut the synthesiser's own leading and trailing silence, so our pauses are exact."""
    hop = TTS_SR // 100
    e = np.array([np.mean(x[i:i + hop] ** 2) for i in range(0, len(x) - hop, hop)])
    db = 10 * np.log10(e + 1e-12)
    on = np.where(db > db.max() + floor_db)[0]
    return x[max(on[0] - 2, 0) * hop:(on[-1] + 3) * hop]


def phone_line(x):
    """Chapter 2's phone line, without the noise: 8 kHz, 300 to 3,400 Hz."""
    down = signal.resample_poly(x, PHONE_SR, TTS_SR)
    band = signal.sosfilt(signal.butter(4, [300, 3400], btype="bandpass", fs=PHONE_SR, output="sos"), down)
    return band


def finish(band, speech_power, rng, snr_db=20):
    """Add line noise at snr_db below the *speech*, so long silences don't change the noise level."""
    noise = rng.normal(size=band.shape) * np.sqrt(speech_power / 10 ** (snr_db / 10))
    out = band + noise
    return (0.9 * out / np.max(np.abs(out))).astype(np.float32)


def main():
    rng = np.random.default_rng(SEED)
    (OUT / "turns").mkdir(parents=True, exist_ok=True)
    (OUT / "bargein").mkdir(parents=True, exist_ok=True)
    rows = []
    for tid, voice, phrases, pauses, kinds in quick(TURNS):
        pieces, segs, t = [np.zeros(int(0.5 * TTS_SR), np.float32)], [], 0.5
        for k, text in enumerate(phrases):
            a = tts(text, voice)
            segs.append({"text": text, "start": round(t, 3), "end": round(t + len(a) / TTS_SR, 3)})
            pieces.append(a)
            t += len(a) / TTS_SR
            if k < len(pauses):
                pieces.append(np.zeros(int(pauses[k] * TTS_SR), np.float32))
                t += pauses[k]
        pieces.append(np.zeros(3 * TTS_SR, np.float32))
        band = phone_line(np.concatenate(pieces))
        speech = np.concatenate([phone_line(p) for p in pieces[1::2]])
        sf.write(OUT / "turns" / f"{tid}.wav", finish(band, np.mean(speech**2), rng), PHONE_SR, subtype="ULAW")
        rows.append({"set": "turns", "id": tid, "voice": voice, "text": " ".join(phrases), "phrases": segs,
                     "pauses": [{"start": segs[k]["end"], "end": segs[k + 1]["start"], "kind": kinds[k]}
                                for k in range(len(pauses))],
                     "turn_end": segs[-1]["end"], "seconds": round(t + 3, 3)})
        print(tid, "done", flush=True)

    prompts = [tts(p, RECEPTIONIST[0]) for p in PROMPTS]
    for i, (kind, text, voice) in enumerate(quick(EVENTS)):
        prompt = prompts[i % len(prompts)]
        x = np.zeros(len(prompt) + 4 * TTS_SR, np.float32)
        # Residual echo: what is left of the receptionist's own voice after the line's echo canceller.
        echo = prompt * 10 ** (-18 / 20)
        x[int(0.5 * TTS_SR):int(0.5 * TTS_SR) + len(prompt)] += echo
        row = {"set": "bargein", "id": f"b{i + 1:02d}", "kind": kind, "text": text, "prompt": PROMPTS[i % len(PROMPTS)],
               "prompt_start": 0.5, "prompt_end": round(0.5 + len(prompt) / TTS_SR, 3)}
        if kind != "none":
            a = tts(text, voice)
            onset = float(rng.uniform(1.5, 3.5))
            x[int(onset * TTS_SR):int(onset * TTS_SR) + len(a)] += a
            row.update(onset=round(onset, 3), offset=round(onset + len(a) / TTS_SR, 3))
            speech_power = np.mean(phone_line(a) ** 2)
        else:
            speech_power = np.mean(phone_line(prompt) ** 2)  # same noise level as the others
        band = phone_line(x)
        sf.write(OUT / "bargein" / f"{row['id']}.wav", finish(band, speech_power, rng), PHONE_SR, subtype="ULAW")
        row["seconds"] = round(len(x) / TTS_SR, 3)
        rows.append(row)
        print(row["id"], kind, "done", flush=True)

    with open(OUT / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
