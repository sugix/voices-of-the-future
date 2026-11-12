---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
date: "2026-12-07"  # quoted: myst-nb stores front matter as JSON
description: Streaming recognition, knowing when a caller has finished, and handling interruptions, measured on hesitant callers and a receptionist that hears its own echo.
---

# Listening live

A caller rings Halfwheel Cycles, digs out the receipt, and says: "Hi, my ticket number is..." Then nothing, for almost a second, while they find the number. A receptionist that waits for silence hears that silence and answers: "Sorry, I didn't catch a ticket number." The caller, mid-breath, says "H W one oh four two" over the top of it, and now two voices are talking at once.

Everything in Chapters 3 and 4 transcribed a recording after it had finished. A live call never finishes on cue. In this chapter I make the receptionist listen *while* the caller talks, and measure the three decisions that makes necessary: how long to wait before writing a word down, when the caller has finished their turn, and what to do when they talk over the receptionist.

Here is where I ended up:

| Finding | Number |
|---|---|
| A model trained on whole recordings needs to hear ahead | Parakeet-TDT with 0.16 s of look-ahead wrote about 6% of the words; it needed about 2 s to get within a few points of its offline accuracy |
| Models built for streaming answer much sooner | Kyutai STT 1B commits to words at least 0.5 s behind the speaker at 7.5% WER on Halfwheel's calls |
| Short silence timeouts cut callers off | with 0.5 s of silence, 14 of 16 hesitating callers were interrupted; with 1.2 s, 3 of 16 |
| The "smart" end-of-turn signals I tried did no better | Kyutai's built-in rule cut off 13 of 16; NVIDIA's end-of-utterance model never fired on phone audio |
| Barge-in is an echo problem first | without echo cancellation, the receptionist stopped for its own voice every time; with it, never, and it caught every real interruption in 0.06 s |
| Duration separates "mm-hm" from "wait, stop" | backchannels lasted 0.70 s or less, interruptions 1.18 s or more |

```{note}
Every number here comes from saved model outputs, so the chapter runs on a CPU in seconds. The capture scripts (`sim/capture_ch5.py`, `sim/aec_ch5.py`) ran on one rented L4 GPU in under an hour; the companion notebook, `chapters/05-listening-live` in [github.com/sugix/voices-of-the-future](https://github.com/sugix/voices-of-the-future), reruns the models on a free Colab T4.
```

## How long to wait before writing a word

A streaming recogniser has to commit to words before the sentence is over. The longer it waits, the more context it has and the fewer mistakes it makes, but the later the receptionist knows what was said. I measured that trade on the same bench as Chapter 4 (Halfwheel's 30 phone calls and 80 LibriSpeech recordings over the phone line), with three kinds of model:

- **Parakeet-TDT 0.6B**, Chapter 4's choice, trained on whole recordings and run in chunks with NVIDIA's own streaming script. Each chunk is decoded with some *look-ahead*, extra audio from the future, so the model waits for chunk plus look-ahead.
- **Parakeet realtime EOU 120M**, a small NVIDIA model trained for streaming, which commits every 160 ms and also predicts the end of an utterance.
- **Kyutai STT**, built for streaming from the start: the 1B model writes each word 0.5 s after hearing it, the 2.6B model 2.5 s after.

```{code-cell} ipython3
import json
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "sim")
from scoring import entity_hits, word_errors  # Chapter 4's scoring
from wav import load  # Chapter 2's WAV reader

refs = {f"halfwheel/{r['id']}": r for r in map(json.loads, open("data/calls/manifest.jsonl"))}
refs |= {f"librispeech/{r['id']}": r for r in map(json.loads, open("data/librispeech/manifest.jsonl"))}
manifest = [json.loads(l) for l in open("data/ch5/manifest.jsonl")]
turns = [r for r in manifest if r["set"] == "turns"]
clips = [r for r in manifest if r["set"] == "bargein"]
nemo = json.load(open("data/ch5/capture_nemo.json"))
kyutai = json.load(open("data/ch5/capture_kyutai.json"))["stt-1b-en_fr"]
kyutai_2b6 = json.load(open("data/ch4/capture_ch4.json"))["results"]["kyutai-stt-2.6b-en"]["files"]


def score(texts):
    """WER on Halfwheel's phone calls and on LibriSpeech over the phone, and Halfwheel entities heard."""
    out = []
    for s in ["halfwheel", "librispeech"]:
        errs = [word_errors(refs[k]["written"], t) for k, t in texts.items() if k.startswith(s)]
        out.append(sum(e for e, _ in errs) / sum(n for _, n in errs))
    hits = [ok for k, t in texts.items() if k.startswith("halfwheel") for _, ok in entity_hits(refs[k], t)]
    return out[0], out[1], sum(hits)


systems = [("Parakeet-TDT 0.6B, whole call at once", np.nan, nemo["tdt offline"]["text"])]
for name, v in nemo.items():
    if name.startswith("tdt chunk"):
        chunk, right = re.findall(r"[\d.]+", name)
        systems.append((f"Parakeet-TDT 0.6B, {chunk} s chunks + {right} s look-ahead", v["latency_s"], v["text"]))
systems.append(("Parakeet realtime EOU 120M", 0.16,
                {k: v["final"].replace("<EOU>", " ") for k, v in nemo["eou120m"]["files"].items()}))
systems.append(("Kyutai STT 1B", 0.5, {k: v["text"] for k, v in kyutai.items()}))
systems.append(("Kyutai STT 2.6B", 2.5, {f"{s}/{i}": v["text"] for key, v in kyutai_2b6.items()
                                          for s, c, i in [key.split("/")] if c == "phone"}))
rows = sorted(((n, lat, *score(t)) for n, lat, t in systems), key=lambda r: (np.nan_to_num(r[1], nan=99), r[0]))
print(f"{'':58} {'waits':>6} {'Halfwheel':>10} {'LibriSpeech':>12} {'entities':>9}")
for n, lat, hw, ls, ent in rows:
    print(f"{n:58} {'all' if np.isnan(lat) else f'{lat:.2f} s':>6} {hw:10.1%} {ls:12.1%} {ent:6}/48")
```
```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 3.6))
for label, pick, style in [("Parakeet-TDT, chunk = look-ahead", lambda n: re.search(r"([\d.]+) s chunks \+ \1 s", n), "o-"),
                           ("Parakeet-TDT, 0.16 s chunks", lambda n: "0.16 s chunks" in n, "s--")]:
    pts = sorted((lat, ls) for n, lat, hw, ls, ent in rows if "TDT" in n and pick(n))
    ax.plot(*zip(*pts), style, label=label)
for n, lat, hw, ls, ent in rows:
    if "TDT" not in n:
        ax.plot(lat, ls, "D", ms=7, label=n)
offline = next(r for r in rows if np.isnan(r[1]))
ax.axhline(offline[3], color="0.5", lw=0.8, ls=":", label="Parakeet-TDT, whole call")
ax.set(xlabel="how long the model waits before committing to a word (s)", ylabel="WER, LibriSpeech by phone",
       ylim=(0, 0.35), title="Accuracy against waiting time")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
```
Three things stand out.

**Look-ahead matters more than chunk size.** With 0.16 s chunks and 0.16 s of look-ahead, Parakeet-TDT wrote about one word in seventeen; for Daniel Okafor's message it produced only "Mm-hmm." With 0.48 s of each, it usually wrote every word but more of them wrong, and on 19 of the 110 recordings it stopped partway (writing under 60% of the words the offline model wrote), at "Could you ask Priya to call me back?" in Daniel's case. Give the same model 0.16 s chunks but 1.84 s of look-ahead and it is back to 9.3%. A model trained on whole recordings has never had to decide about a word without hearing what comes after it, and it falls apart when forced to. Even at 4 s it is not quite back to its whole-call accuracy.

**Models trained to stream are far better at low delay.** Kyutai's 1B model is the best of the low-delay options on Halfwheel's calls (7.5%), though it struggles more with LibriSpeech's older, more literary English (15.7%). NVIDIA's 120M model answers fastest but is the least accurate, and it writes numbers the way they sound to it ("four hundred and fifteen"), so the receptionist would need to turn those back into digits.

**Nothing streaming beats waiting for the whole turn.** That suggests a different design. Watch when each word appears for the first caller:

```{code-cell} ipython3
r = turns[0]
key = f"turns/{r['id']}"
print("What the caller said, and when:")
for p in r["phrases"]:
    print(f"  {p['start']:5.2f} to {p['end']:5.2f} s   {p['text']}")
print("\nWhen each streaming model had written each word:")


def arrivals(growing):
    """[(time, text so far)] -> each final word with the time its first letters appeared."""
    first = []
    for t, text in growing:
        first += [t] * (len(text.split()) - len(first))
    return list(zip(first, growing[-1][1].split()))


kyutai_growing, text = [], ""
for t, piece in kyutai[key]["pieces"]:
    text += piece
    kyutai_growing.append((t, text))
for name, growing in [("Kyutai 1B", kyutai_growing), ("EOU 120M", [(t, x.replace("<EOU>", "")) for t, x in nemo["eou120m"]["files"][key]["events"]])]:
    print(f"  {name:10}" + "  ".join(f"{w} {t:.2f}" for t, w in arrivals(growing)))
```
Kyutai writes "is" at 2.96 s, a second after the caller said it, while NVIDIA's model writes it almost immediately but later turns "one oh four two" into "one hundred and four two". The receptionist does not need the words while the caller is still talking; it needs them, correct, the moment the caller *stops*. On a GPU, Parakeet-TDT transcribes a whole Halfwheel call in about a tenth of a second (Chapter 4). So my design for Halfwheel is: listen continuously to decide *when* the caller has finished, then transcribe the finished turn in one go with the accurate model. That makes the next question, when has the caller finished, the one that decides how responsive the receptionist feels.

## Has the caller finished?

People in conversation leave remarkably short gaps between turns, around 200 ms on average across many languages {cite}`stivers2009turntaking`, because they predict the end of a turn from its words and rhythm rather than waiting for silence. A machine that waits for silence has to choose how much silence means "done".

To measure that, I need callers who pause *mid-turn*. The script `sim/turn_calls.py` synthesises 16 turns phrase by phrase with Kokoro and puts real pauses inside them: after "my ticket number is", while reading out a phone number, before "um, Trek Domane". Because every phrase is synthesised separately, I know exactly when each one starts and ends, and when the turn really ends. My pauses are deliberately long, half a second to 1.4 s, the hesitations of someone reading off a receipt or remembering a model name. That is the hard case, and it is common on a bike shop's phone line.

The simplest rule is a silence timeout: run Silero VAD (Chapter 2) and end the turn after a fixed amount of silence.

```{code-cell} ipython3
SILERO, FRAME = nemo["silero"]["probs"], nemo["silero"]["frame_s"]


def silence_endpoint(key, wait, start=0.7):
    """End the turn once Silero has heard no speech for `wait` seconds."""
    quiet = 0.0
    for i, p in enumerate(SILERO[key]):
        t = (i + 1) * FRAME
        quiet = 0.0 if p > 0.5 else quiet + FRAME
        if t > start and quiet >= wait - 1e-9:
            return t
    return None


def evaluate(endpoint):
    """How many turns were cut off mid-sentence, and how long the rest waited after the caller finished."""
    cut, waits = [], []
    for r in turns:
        t = endpoint(f"turns/{r['id']}")
        if t is not None and t < r["turn_end"]:
            cut.append(r["id"])
        elif t is not None:
            waits.append(t - r["turn_end"])
    return len(cut), np.median(waits) if waits else np.nan, cut


pauses = [p["end"] - p["start"] for r in turns for p in r["pauses"]]
print(f"{len(turns)} turns, {len(pauses)} mid-turn pauses from {min(pauses):.1f} to {max(pauses):.1f} s\n")
print(f"{'rule':38} {'cut off':>8} {'then waits':>11}")
for wait in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]:
    n, w, _ = evaluate(lambda k, wait=wait: silence_endpoint(k, wait))
    print(f"{f'Silero, {wait} s of silence':38} {n:5}/16 {w:10.2f} s")
```
There is no free setting. A snappy 0.5 s timeout interrupts 14 of the 16 callers. Waiting 1.2 s interrupts 3 and adds 1.2 s to every reply. I tried three smarter signals:

- **Kyutai's end-of-turn heads.** The 1B model has four small extra outputs that predict whether there will be no speech in the next 0.5, 1, 2 and 3 seconds; Kyutai's own tools end the turn when the 2-second head passes 0.5. The PyTorch release ships without these heads, but Kyutai's Rust release has them and is otherwise identical tensor for tensor, so the capture script attaches them.
- **NVIDIA's end-of-utterance token**, which the 120M model writes into its transcript as `<EOU>`.
- **A rule of my own**: after 0.5 s of silence, end the turn unless the transcript so far ends in a word that cannot end a sentence ("is", "my", "the", "for"), in which case wait up to 1.5 s.

```{code-cell} ipython3
def kyutai_endpoint(key, threshold=0.5, head=2, start=0.7):
    """Kyutai's rule: end the turn when head 2 (no speech within 2 s) passes the threshold."""
    for t, *heads in kyutai[key]["prs"]:
        if t > start and heads[head] > threshold:
            return t
    return None


def text_at(key, t):
    return "".join(w for tt, w in kyutai[key]["pieces"] if tt <= t)


UNFINISHED = set("a an the to for of on in at my is are and or but with from by your it's i'm".split())


def last_word_endpoint(key, short=0.5, long=1.5, start=0.7):
    """Wait `short` seconds, unless the transcript so far ends in a word that cannot end a sentence."""
    quiet = 0.0
    for i, p in enumerate(SILERO[key]):
        t = (i + 1) * FRAME
        quiet = 0.0 if p > 0.5 else quiet + FRAME
        if t <= start or quiet < short - 1e-9:
            continue
        words = re.findall(r"[a-z']+", text_at(key, t).lower())
        if (words and words[-1] not in UNFINISHED) or quiet >= long - 1e-9:
            return t
    return None


eou = lambda key: next((t for t, txt in nemo["eou120m"]["files"][key]["events"] if "<EOU>" in txt), None)
print(f"{'rule':38} {'cut off':>8} {'then waits':>11}")
for name, fn in [("Kyutai head 2 > 0.5 (Kyutai's default)", kyutai_endpoint),
                 ("Kyutai head 2 > 0.9", lambda k: kyutai_endpoint(k, 0.9)),
                 ("last word, 0.5 s or 1.5 s", last_word_endpoint)]:
    n, w, _ = evaluate(fn)
    print(f"{name:38} {n:5}/16 {w:10.2f} s")
late = sum(eou(f"turns/{r['id']}") is None or eou(f"turns/{r['id']}") > r["seconds"] - 0.4 for r in turns)
print(f"{'Parakeet EOU token':38} {'0':>5}/16   only when the audio ended, in {late}/16 turns")
r = turns[0]
gap = r["pauses"][0]
print(f"\n{r['id']}: pause from {gap['start']:.2f} to {gap['end']:.2f} s. Transcript after 0.5 s of it: "
      f"'{text_at('turns/t01', gap['start'] + 0.5).strip()}'")
```
None of them beat a plain 1.2 s timeout here.

Kyutai's heads fire within a fraction of a second of almost every pause, including the unfinished ones; in this data they behave like a fast pause detector, not a judge of whether the sentence is complete. Raising the threshold to 0.9 helps, but then it waits longer than the timeout does. NVIDIA's model never produced `<EOU>` until the audio stopped altogether. In a separate check on clean, full-band recordings it did fire, 0.5 to 2 s after the speaker stopped, so I suspect the phone line's constant hiss; either way, on Halfwheel's audio it is not usable as it stands.

My own rule failed for the most instructive reason. Look at the last line: half a second into the pause after "my ticket number is", the streaming transcript still read "Hi, my ticket number". The word "is" had not arrived yet. Any rule that reads the transcript is reading a transcript that lags the audio, and it makes its decision about words the caller said a second ago. Here is that call, with where each rule ended the turn:

```{code-cell} ipython3
r = turns[0]
key = f"turns/{r['id']}"
x, sr = load(f"data/ch5/turns/{r['id']}.wav")
fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
ax[0].plot(np.arange(len(x)) / sr, x, lw=0.3, color="0.4")
for p in r["phrases"]:
    ax[0].text(p["start"], 0.95, p["text"], fontsize=7, va="top")
ax[0].set(title=f"{r['id']}: a caller reading a ticket number off a receipt", ylim=(-1, 1.1), yticks=[])
t_sil = (np.arange(len(SILERO[key])) + 1) * FRAME
ax[1].plot(t_sil, SILERO[key], lw=1, label="Silero: speech now")
prs = np.array(kyutai[key]["prs"])
ax[1].plot(prs[:, 0], prs[:, 3], lw=1, label="Kyutai head 2: silence ahead")
for name, t, c, y in [("Kyutai > 0.5", kyutai_endpoint(key), "C3", 0.85), ("Silero 0.5 s", silence_endpoint(key, 0.5), "C4", 0.6),
                      ("Silero 1.2 s", silence_endpoint(key, 1.2), "C2", 0.35)]:
    for a in ax:
        a.axvline(t, color=c, ls="--", lw=1)
    ax[1].text(t + 0.04, y, f"ends turn: {name}", color=c, fontsize=7)
ax[1].set(xlabel="seconds", ylabel="probability", ylim=(0, 1.05))
ax[1].legend(fontsize=7, loc="lower right")
fig.tight_layout()
```
Kyutai's head (orange) rises almost as soon as the caller stops for breath, and Silero with 0.5 s of patience ends the turn before "H W one oh four two" begins. Only the 1.2 s timeout waits for the number.

For Halfwheel I take **Silero with a 1.2 s timeout**, and accept that it will sometimes still jump in, as it did for 3 of the 16 callers. That only works if jumping in is cheap to recover from: when the caller carries on talking, the receptionist must notice and stop. That is the last problem in this chapter.

## When the caller talks over the receptionist

Callers interrupt. "Sorry, it's actually Friday." "Wait, wrong bike." A good receptionist stops talking at once. Callers also say "mm-hm" and "okay" while listening, and a good receptionist carries on. The script also builds 12 clips of the receptionist reading a reply, in a Kokoro voice no caller uses (`bf_lily`), while the caller interrupts (4 clips), murmurs a backchannel (4) or stays silent (4).

There is a catch. On a real phone call, some of the receptionist's own voice comes back down the line as **echo**. Phone networks and handsets cancel most of it, but not all, especially on speakerphone. My clips leave the echo 18 dB below the receptionist's voice, about what a weak echo canceller on a speakerphone might leave behind. Here is what happens if the receptionist stops whenever Silero hears a voice:

```{code-cell} ipython3
def voice_stop(probs, r, sustain=0.0):
    """Stop talking at the first moment Silero has heard `sustain` seconds of voice during our own speech."""
    run = 0.0
    for i, p in enumerate(probs):
        t = (i + 1) * FRAME
        run = run + FRAME if p > 0.5 else 0.0
        if r["prompt_start"] < t <= r["prompt_end"] and run >= max(sustain, FRAME) - 1e-9:
            return t
    return None


def barge_report(name, stop):
    caught, early, false = [], 0, {"backchannel": 0, "none": 0}
    for r in clips:
        t = stop(r)
        if r["kind"] == "barge-in" and t is not None:
            if t >= r["onset"]:
                caught.append(t - r["onset"])
            else:
                early += 1
        elif r["kind"] != "barge-in" and t is not None:
            false[r["kind"]] += 1
    delay = f"{np.median(caught):.2f} s" if caught else "-"
    print(f"{name:30} {len(caught):3}/4 {early:9}/4 {delay:>8} {false['backchannel']:8}/4 {false['none']:7}/4")


header = f"{'stop when':30} {'caught':>6} {'too early':>11} {'delay':>8} {'mm-hm':>10} {'echo':>9}"
print(header)
for name, s in [("any voice", 0.0), ("300 ms of voice", 0.3), ("600 ms of voice", 0.6)]:
    barge_report(name, lambda r, s=s: voice_stop(SILERO[f"bargein/{r['id']}"], r, s))
print("\nWhat Kyutai 1B heard while the receptionist was talking, with nobody else on the line:")
for r in clips:
    if r["kind"] == "none":
        print(f"  said:  {r['prompt'][:70]}\n  heard: {kyutai['bargein/' + r['id']]['text'][:70]}")
        break
```
It stops every single time, before anyone has interrupted. The voice it hears is its own. Requiring a longer stretch of voice does not help, because its own speech is continuous, and the transcript does not help either: Kyutai faithfully transcribed the echo, as "Castle and Cycles".

The fix is the one telephony systems use: **echo cancellation**. The receptionist knows exactly what it is playing, so it can learn how that signal comes back down the line and subtract it from what it hears. The classic method is an adaptive filter called **normalised least mean squares** (NLMS): predict the echo from the last few milliseconds of what was played, subtract the prediction, and nudge the filter towards whatever was left over. It is a dozen lines:

```{code-cell} ipython3
def cancel_echo(mic, ref, taps=32, mu=0.1, eps=1e-3):
    """Normalised LMS: learn how our own voice comes back down the line, and subtract it."""
    w = np.zeros(taps)
    pad = np.concatenate([np.zeros(taps - 1), ref])
    out = np.empty_like(mic)
    for n in range(len(mic)):
        x = pad[n:n + taps][::-1]  # the last `taps` samples we played, newest first
        e = mic[n] - w @ x  # what we hear, minus our prediction of our own echo
        w += mu * e * x / (x @ x + eps)  # nudge the echo model towards what we heard
        out[n] = e
    return out


def reference_for(r, n):
    """What the receptionist was playing, lined up with the call recording."""
    p, sr = load(f"data/ch5/prompts/p{(int(r['id'][1:]) - 1) % 4 + 1}.wav")
    ref = np.zeros(n, dtype=np.float32)
    i = int(r["prompt_start"] * sr)
    ref[i:i + len(p)] = p[: n - i]
    return ref


# --- demo: the four clips where nobody interrupts, so everything we hear during the prompt is echo
for r in [c for c in clips if c["kind"] == "none"]:
    mic, sr = load(f"data/ch5/bargein/{r['id']}.wav")
    clean = cancel_echo(mic, reference_for(r, len(mic)))
    span = slice(int((r["prompt_start"] + 1.0) * sr), int(r["prompt_end"] * sr))  # after a second of learning
    hiss = slice(int((r["prompt_end"] + 0.5) * sr), None)  # the open line once we stop talking
    db = lambda x: 10 * np.log10(np.mean(x**2))
    print(f"{r['id']}: while we talk {db(mic[span]):6.1f} dB, after cancelling {db(clean[span]):6.1f} dB, "
          f"line hiss alone {db(mic[hiss]):6.1f} dB")
```
On the four clips where nobody interrupts, cancelling brings the level during the receptionist's speech down to within about half a decibel of the line hiss that remains once it stops talking: the echo is essentially gone. (I tuned the filter on these clips, choosing 32 taps and a small step size of 0.1; larger filters and steps adapted faster but left more residue on this noisy line.) Real systems use more robust cancellers, such as the one in WebRTC or the one your telephony provider runs, but the idea is the same. Here is the same barge-in test on the cleaned audio:

```{code-cell} ipython3
aec = json.load(open("data/ch5/capture_aec.json"))["probs"]
print(header)
for name, s in [("any voice", 0.0), ("300 ms of voice", 0.3), ("600 ms of voice", 0.6)]:
    barge_report(name, lambda r, s=s: voice_stop(aec[f"bargein/{r['id']}"], r, s))


print("\nHow long Silero heard a voice, once the echo was gone:")
for r in clips:
    if r["kind"] != "none":
        p = np.array(aec[f"bargein/{r['id']}"]) > 0.5
        t = (np.arange(len(p)) + 1) * FRAME
        heard = p[(t > r["onset"] - 0.1) & (t < r["offset"] + 0.6)].sum() * FRAME
        print(f"  {r['kind']:12} {r['text']:42} {heard:.2f} s")
```
Now "any voice" catches all four interruptions within 0.06 s and never stops for its own echo. It still stops for every "mm-hm", and it should hear them: they are a real voice. The difference between a backchannel and an interruption is not *whether* the caller spoke but how much they said, and the last table shows the split clearly in this data: backchannels lasted 0.70 s or less, interruptions 1.18 s or more.

So the receptionist's barge-in rule is two steps. On any voice, **pause** immediately (within 0.06 s). If the voice is still going after about 0.9 s, **stop** and listen; if it has already ended, **resume** where it paused. A short, urgent "Stop!" would slip under that threshold; checking the transcript for words like "stop", "wait" and "no" once it arrives would catch it (exercise 4).

## Halfwheel's live listening

Putting the chapter together, the receptionist listens like this:

1. Cancel its own echo from the incoming audio.
2. Run Silero VAD on 32 ms frames.
3. While the caller speaks, do nothing but listen. Once Silero has heard 1.2 s of silence, the turn is over.
4. Transcribe the finished turn with Parakeet-TDT, boosting and snapping as in Chapter 4: about 0.1 s.
5. While the receptionist replies, pause on any voice, and stop if it lasts more than about 0.9 s.

That is 1.3 s from the caller finishing to the receptionist knowing what they said, before it has thought of an answer or spoken a word. Chapter 8 measures the whole wait end to end, and Chapter 10 looks at models that skip the transcript entirely.

## When it breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| Streaming transcript is empty or stops after the first sentence | An offline model run with too little look-ahead | Give it 1.5 to 2 s of look-ahead, or use a model trained to stream |
| The receptionist answers while the caller is still reading out a number | Silence timeout too short | Lengthen it (1.2 s here), and make interruptions cheap to recover from |
| An "end of turn" model fires on every breath | It predicts pauses, not finished thoughts | Test it on your own callers' hesitations before trusting it |
| Rules based on the transcript decide too early | The streaming transcript lags the audio by half a second or more | Decide timing from audio; use the transcript afterwards |
| The receptionist stops mid-sentence for no reason | It hears its own echo | Cancel echo with the playback signal as reference before any voice detection |
| It stops for every "mm-hm" | Treating any voice as an interruption | Pause on voice, stop only if it lasts; check the words |

```{tip}
**Takeaways**
- Streaming is a trade between waiting and accuracy, and models trained on whole recordings trade badly. Measure word error rate at the delay you will actually run.
- You often do not need words while the caller is talking. Decide *when* they have finished from the audio, then transcribe the whole turn accurately.
- Silence is a poor signal of a finished turn, but on hesitant phone callers it beat every smarter signal I tried. Expect to wait about a second, and plan for interrupting by mistake.
- Before detecting a caller's voice, remove your own. Echo cancellation turns barge-in from impossible into easy.
- Pause on any voice, stop on a long one: that separates "mm-hm" from "wait, stop".
```

## Try it yourself

The companion notebook, `chapters/05-listening-live`, has the saved outputs, every cell above and the capture scripts.

1. **Your own hesitations.** Add three turns of your own to `TURNS` in `sim/turn_calls.py`, with pauses where you naturally hesitate, and rerun the endpoint table. Does 1.2 s still work?
2. **Combine the signals.** End the turn after 0.8 s of silence only if Kyutai's 3-second head is also above 0.9, otherwise after 1.5 s. Does it beat the plain timeout?
3. **Break the echo canceller.** Raise `mu` to 1.0, or set `taps` to 4. Watch the residue rise and the false barge-ins return.
4. **Shorter interruptions.** Add "Stop!" and "No." as barge-ins. What does the 0.9 s rule do with them, and what would fix it?

## Next

The receptionist can now hear a caller properly: it knows when they are talking, when they have finished, and when they want it to be quiet. It still has nothing to say back. Chapter 6 opens the other half of the stack: how modern text-to-speech models turn the receptionist's reply into a voice.
