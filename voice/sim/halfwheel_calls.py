"""Generate the Halfwheel Cycles sample calls used throughout the book.

Halfwheel Cycles is a fictional bike repair shop. Every call is synthesised
with Kokoro-82M (Apache-2.0), so readers can regenerate the exact same files.
Each call is written twice:

  data/calls/clean/<id>.wav   24 kHz, as the TTS model produced it
  data/calls/phone/<id>.wav   8 kHz G.711 mu-law, band-limited, with seeded
                              background noise, the way a phone line delivers it

plus data/calls/manifest.jsonl with, for every call, the spoken text (what the
TTS model was asked to say), the written text (how the receptionist should
write it down) and the entities that matter to the shop: tickets, bike models,
names and phone numbers.

The first six calls are hand-written and used in Chapter 3. The rest are drawn
from templates with a fixed seed. Existing clean files are reused, so adding
calls never changes the earlier ones.

Run on the pod (CPU is fine):
  python voice_applications/sim/halfwheel_calls.py
Needs: kokoro>=0.9.2, soundfile, scipy, numpy, and the espeak-ng system package.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SEED = 1042
TTS_SR = 24_000
PHONE_SR = 8_000
OUT = Path(__file__).resolve().parents[1] / "data" / "calls"
N_GENERATED = 24


@dataclass
class Call:
    id: str
    voice: str  # Kokoro voice name
    lang: str   # Kokoro lang code: "a" American English, "b" British English
    snr_db: float  # background noise level on the phone version
    text: str  # what the TTS model says (spoken form)
    kind: str  # what the receptionist has to do with it
    written: str = ""  # how the receptionist should write it down, if different
    entities: list = field(default_factory=list)  # [{"type", "written"}]


def ent(kind: str, written: str) -> dict:
    return {"type": kind, "written": written}


# Chapter 3's calls. Kept exactly as they were.
CALLS = [
    Call("c01_status", "af_heart", "a", 25,
         "Hi, is my bike ready? The ticket number is H W one oh four two.",
         "repair_status", "Hi, is my bike ready? The ticket number is HW-1042.",
         [ent("ticket", "HW-1042")]),
    Call("c02_booking", "am_michael", "a", 20,
         "Hello, I'd like to book a tune-up for my Trek Domane S L five, "
         "sometime Thursday afternoon if you have it.",
         "booking", "Hello, I'd like to book a tune-up for my Trek Domane SL 5, "
         "sometime Thursday afternoon if you have it.",
         [ent("bike", "Trek Domane SL 5")]),
    Call("c03_price", "bf_emma", "b", 22,
         "How much do you charge to true a rear wheel and replace the brake pads?",
         "faq"),
    Call("c04_message", "bm_george", "b", 18,
         "Could you ask Priya to call me back? It's Daniel Okafor, "
         "on four one five, five five five, oh one nine two.",
         "message", "Could you ask Priya to call me back? It's Daniel Okafor, "
         "on 415-555-0192.",
         [ent("name", "Daniel Okafor"), ent("phone", "415-555-0192")]),
    Call("c05_change", "af_bella", "a", 15,
         "I booked a service for Saturday at ten, but I need to move it to Monday.",
         "booking_change"),
    Call("c06_spam", "am_adam", "a", 20,
         "Congratulations! Your business has been selected for a limited time "
         "merchant processing offer. Press one now to speak with a specialist.",
         "spam"),
]

# Material for the generated calls. Bike models are real products, named only
# as things a customer would own; (written, spoken) pairs where they differ.
BIKES = [("Trek Domane SL 5", "Trek Domane S L five"),
         ("Specialized Tarmac SL7", "Specialized Tarmac S L seven"),
         ("Cannondale Synapse", None), ("Brompton C Line", None),
         ("Giant Defy Advanced 2", "Giant Defy Advanced two"),
         ("Canyon Grizl", "Canyon Grizzle"), ("Surly Cross-Check", "Surly Cross Check"),
         ("Bianchi Oltre", None), ("Orbea Orca", None), ("Ribble Endurance SL", "Ribble Endurance S L")]
NAMES = ["Daniel Okafor", "Siobhan Murphy", "Tomasz Wojcik", "Mei Nguyen", "Raj Patel",
         "Grace Adeyemi", "Liam O'Sullivan", "Sofia Rossi", "Hannah Schmidt", "Kwame Mensah",
         "Aoife Byrne", "Diego Alvarez"]
STAFF = ["Priya", "Marco", "Jess"]
SERVICES = ["tune-up", "wheel true", "brake bleed", "drivetrain clean", "new chain and cassette"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
AREA = ["415", "312", "646", "206", "617"]
VOICES = [("af_heart", "a"), ("am_michael", "a"), ("bf_emma", "b"), ("bm_george", "b"),
          ("af_bella", "a"), ("af_nicole", "a"), ("am_fenrir", "a"), ("bf_isabella", "b"),
          ("bm_fable", "b")]
DIGIT = {"0": "oh", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
         "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def say_digits(s: str) -> str:
    return " ".join(DIGIT[c] for c in s)


def generated_calls(n: int, seed: int) -> list[Call]:
    rng = np.random.default_rng(seed)
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    calls = []
    for i in range(n):
        voice, lang = pick(VOICES)
        snr = float(rng.choice([10, 15, 20, 25]))
        kind = ["repair_status", "booking", "message", "faq", "booking_change", "spam"][i % 6]
        ticket = f"HW-{int(rng.integers(1000, 2000))}"
        t_spoken = "H W " + say_digits(ticket[3:])
        bike, b_spoken = pick(BIKES)
        b_spoken = b_spoken or bike
        name = pick(NAMES)
        phone = f"{pick(AREA)}-555-01{int(rng.integers(0, 100)):02d}"
        p_spoken = ", ".join(say_digits(part) for part in phone.split("-"))
        day, service, staff = pick(DAYS), pick(SERVICES), pick(STAFF)
        if kind == "repair_status":
            s = f"Hi, it's {name}. Is my {{b}} ready yet? The ticket is {{t}}."
            ents = [ent("name", name), ent("bike", bike), ent("ticket", ticket)]
        elif kind == "booking":
            s = f"Hello, could I book a {service} for my {{b}} on {day} morning? My name's {name}."
            ents = [ent("bike", bike), ent("name", name)]
        elif kind == "message":
            s = f"Can you ask {staff} to call {name} back on {{p}}? It's about the {{b}}."
            ents = [ent("name", name), ent("phone", phone), ent("bike", bike)]
        elif kind == "faq":
            s = f"How much would a {service} cost on a {{b}}, and how long does it take?"
            ents = [ent("bike", bike)]
        elif kind == "booking_change":
            s = f"This is {name}, ticket {{t}}. Could I pick the bike up on {day} instead?"
            ents = [ent("name", name), ent("ticket", ticket)]
        else:
            s = pick(["Final notice about your vehicle's extended warranty. Press nine to be removed "
                      "or press one to speak with an agent.",
                      "Hi, this is Ashley from business funding services. You qualify for a "
                      "same day loan of up to fifty thousand dollars. Call us back today.",
                      "This call is about your Google business listing. It will be suspended "
                      "unless you press one now to verify it."])
            ents = []
        spoken = s.format(b=b_spoken, t=t_spoken, p=p_spoken)
        written = s.format(b=bike, t=ticket, p=phone)
        calls.append(Call(f"g{i + 1:02d}_{kind}", voice, lang, snr, spoken, kind,
                          written if written != spoken else "", ents))
    return calls


def to_phone(audio: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate a phone line: 8 kHz, 300-3400 Hz band, background noise."""
    down = signal.resample_poly(audio, PHONE_SR, sr)
    sos = signal.butter(4, [300, 3400], btype="bandpass", fs=PHONE_SR, output="sos")
    band = signal.sosfilt(sos, down)
    noise = rng.normal(0.0, 1.0, size=band.shape)
    p_sig = np.mean(band**2)
    noise *= np.sqrt(p_sig / (10 ** (snr_db / 10)) / np.mean(noise**2))
    out = band + noise
    return (0.9 * out / np.max(np.abs(out))).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(SEED)  # phone-line noise, consumed in call order
    (OUT / "clean").mkdir(parents=True, exist_ok=True)
    (OUT / "phone").mkdir(parents=True, exist_ok=True)
    pipelines = {}
    rows = []
    for call in CALLS + generated_calls(N_GENERATED, SEED + 1):
        clean_path = OUT / "clean" / f"{call.id}.wav"
        if clean_path.exists():
            audio, _ = sf.read(clean_path, dtype="float32")
        else:
            from kokoro import KPipeline  # imported lazily so reruns need no TTS
            pipe = pipelines.get(call.lang) or pipelines.setdefault(call.lang, KPipeline(lang_code=call.lang))
            chunks = [a for _, _, a in pipe(call.text, voice=call.voice)]
            audio = np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks])
            sf.write(clean_path, audio, TTS_SR, subtype="PCM_16")
            audio, _ = sf.read(clean_path, dtype="float32")  # same samples as a rerun sees
        phone = to_phone(audio, TTS_SR, call.snr_db, rng)  # always run: keeps the noise stream aligned
        phone_path = OUT / "phone" / f"{call.id}.wav"
        if not phone_path.exists():
            # G.711 mu-law, the encoding most phone networks carry.
            sf.write(phone_path, phone, PHONE_SR, subtype="ULAW")
        row = asdict(call)
        row["written"] = call.written or call.text
        rows.append({**row, "seconds": round(len(audio) / TTS_SR, 2)})
        print(f"{call.id}: {rows[-1]['seconds']} s", flush=True)
    with open(OUT / "manifest.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
