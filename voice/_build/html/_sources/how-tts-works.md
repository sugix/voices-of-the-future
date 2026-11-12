---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
date: "2026-12-14"  # quoted: myst-nb stores front matter as JSON
description: Four open text-to-speech models read Halfwheel's receptionist script. I measure which facts they get wrong, how soon they start talking, and what survives the phone line.
---

# How modern TTS works

A caller rings Halfwheel Cycles to ask whether Priya can ring them back. The language model behind the receptionist writes a perfectly good reply: "Priya will call you back on 415-555-0192 before 6pm." Then a text-to-speech model reads it out loud, and the caller hears "four hundred fifteen, five fifty five, one ninety two" and writes down a number that doesn't exist.

Chapters 2 to 5 were about listening. This chapter is about the other half: how the receptionist talks. I gave four open TTS models the same twelve lines the receptionist says every day, ran every clip back through Chapter 4's recogniser to check what a listener would hear, and timed how long each model took to start speaking.

Here is where I ended up:

| Finding | Number |
|---|---|
| Text written for a screen trips up every model | read as written, the models got between 11 and 15 of the 22 facts right (ticket, prices, times, phone number, spellings) |
| A page of rules fixes nearly all of it | after rewriting each line the way a receptionist would say it, Pocket TTS and Kyutai TTS got 65 of 66 facts right over three reads each |
| Speech needs surprisingly few bits | 8 of Mimi's 32 codebooks (1.1 kbit/s) kept word error rate at 3.8%, against 2.3% for the original recordings; a phone line spends 64 kbit/s |
| How soon a model starts talking varies a hundredfold | Kokoro on a GPU 0.09 s, Pocket TTS on two CPU threads 0.19 s, Kyutai TTS 1.6B on an L4 0.96 s, CSM-1B 11 s |
| Sampled models need guard rails | CSM-1B ran on past 20 s on 4 of its 48 clips and got 54 of 66 facts right |
| The phone line costs synthetic speech almost nothing | over the simulated phone line, Pocket TTS and Kyutai TTS got all 66 facts right |

```{note}
Every number here comes from saved model outputs, so the chapter runs on a CPU in seconds. The capture script, `sim/capture_ch6.py`, ran all four models on one rented L4 GPU in about 35 minutes; the companion notebook, `chapters/06-how-tts-works` in [github.com/sugix/voices-of-the-future](https://github.com/sugix/voices-of-the-future), reruns them on a free Colab T4, and Kokoro and Pocket TTS run on a laptop CPU.
```

## One line, four voices

The four models cover the three designs you will meet in open TTS today:

- **Kokoro-82M** turns text into phonemes and phonemes into sound. Small, fast and predictable; Chapter 5's receptionist already speaks with its `bf_lily` voice.
- **Pocket TTS**, Kyutai's 100M model, built to run on a laptop CPU.
- **Kyutai TTS 1.6B**, a streaming model that generates audio tokens.
- **CSM-1B** from Sesame, a conversational model that also generates audio tokens.

Pocket TTS, Kyutai TTS and CSM can all copy a voice from a short recording, so I gave them the same speaker: VCTK speaker p228, a young woman from southern England. Kokoro has a fixed set of voices and keeps `bf_lily`.

This is the script, written the way a language model writes it:

```{code-cell} ipython3
# The receptionist's script: twelve things Halfwheel's receptionist says every day, written the way
# a language model writes them, with the shop's tickets, prices, times and names.
LINES = [
    ("r01", "Thanks for calling Halfwheel Cycles, you're through to the workshop. How can I help?"),
    ("r02", "Your ticket is HW-1042, and the Trek Domane is ready to collect after 2pm."),
    ("r03", "A wheel true is £25, and a full service is £149.99."),
    ("r04", "I've booked you in for Thu 14 Nov at 10:30."),
    ("r05", "Priya will call you back on 415-555-0192 before 6pm."),
    ("r06", "Is that Okafor, O-K-A-F-O-R?"),
    ("r07", "We've ordered a Shimano 105 Di2 rear mech, so allow 3-5 working days."),
    ("r08", "We're open Mon-Sat, 9am-6pm."),
    ("r09", "That comes to £37.50 including VAT."),
    ("r10", "Sorry, the Canyon Grizl won't be ready until Friday, but the Brompton is done."),
    ("r11", "Your next service is due at 1,500 miles."),
    ("r12", "Mm, let me just check that for you. Okay, yes, I can see it here."),
]
```

And this is the line from the opening, read by each model exactly as written, with what Parakeet-TDT heard when I played the audio back to it:

```{code-cell} ipython3
import json
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display

sys.path.insert(0, "sim")
from scoring import edit_ops, normalize, word_errors  # Chapter 4's scoring
from wav import load  # Chapter 2's WAV reader

MODELS = {"kokoro": "Kokoro-82M", "pocket": "Pocket TTS", "kyutai": "Kyutai TTS 1.6B", "csm": "CSM-1B"}
synth = {}
for m in MODELS:
    synth |= json.load(open(f"data/ch6/synth_{m}.json"))["clips"]
heard = json.load(open("data/ch6/capture_asr.json"))  # Parakeet-TDT's transcript of every clip


def play(name):
    x, sr = load(f"data/ch6/listen/{name}.wav")
    display(Audio(x, rate=sr))


print(f"{'Written':16s} {dict(LINES)['r05']}")
for m, name in MODELS.items():
    print(f"{name:16s} {heard[f'{m}/r05_raw_0']['wide']}")
    play(f"{m}_r05_raw")
```

Every model got something wrong, and each got it wrong differently. Kokoro read the phone number as three numbers, "four hundred fifteen, five fifty five, one ninety two", and the zero vanished. Pocket TTS said the area code as "four fifteen" and added a five. Kyutai read the number perfectly, then dropped "before 6pm". CSM read the number back as something Parakeet wrote down as "455-55DR19192". None of these is a broken model. They are models being asked to read text that was written for a screen.

## Three ways to turn text into sound

Every model here does the same job: text goes in, a waveform comes out. They differ in what they predict in between, and that one choice explains most of what I measured later.

**Phonemes, then a vocoder (Kokoro).** Kokoro-82M is built on StyleTTS 2 {cite}`li2023styletts2`. A grapheme-to-phoneme front end (misaki, with espeak-ng as a fallback) turns the text into phonemes. A duration predictor decides how long each one lasts, a decoder paints the whole utterance at once, and an iSTFTNet vocoder turns it into 24 kHz audio. Nothing is sampled and nothing is generated one step at a time, which is why it is fast and why it says a sentence the same way every time. It is also why its mistakes are easy to find: the phonemes are a readable intermediate, and when the audio is wrong, the phonemes were wrong first.

```{code-cell} ipython3
for lid in ["r02", "r05", "r08"]:
    print(dict(LINES)[lid])
    print("   ", synth[f"kokoro/{lid}_raw_0"]["phonemes"], "\n")
```

You don't need to read IPA to see the problems. In the ticket, `ˌAʧdˈʌbᵊljuː tˈɛn fˈɔːti tˈuː` is "aitch double-u ten forty two", where Halfwheel's tickets are read digit by digit. The phone number is `fˈɔː hˈʌndɹəd fˌɪftˈiːn...`, "four hundred fifteen". And "Mon-Sat" has become one word, `mˈQnsˌat`, "monsat". Kokoro's front end made sensible guesses for a general reader. It doesn't know it works for a bike shop.

**Audio tokens from a codec language model (Kyutai TTS, CSM).** Kyutai STT, the streaming recogniser from Chapters 4 and 5, never sees a waveform: it hears through Mimi, Kyutai's neural codec. Mimi squeezes 24 kHz audio into 12.5 frames per second, each frame a stack of up to 32 codes from a residual vector quantiser. The first code in each frame carries most of what was said, and every further code refines what the earlier ones left out. Once audio is a sequence of tokens, generating speech looks like generating text: a transformer predicts the next frame of codes, and Mimi's decoder turns codes back into sound.

Both token models split that work the same way. A large backbone transformer looks at the text and all the audio so far, once per frame; a small depth transformer then predicts that frame's 32 codes one after another. CSM-1B uses a 1B Llama backbone and a 100M depth decoder. Kyutai TTS 1.6B uses a 1B backbone and adds the idea from its paper, *delayed streams modelling* {cite}`zeghidour2025dsm`: text and audio are two time-aligned streams, and the audio runs 16 frames (1.28 s) behind the text, so the model always knows the next few words before it has to say them. Both models sample, so they read the same sentence a little differently every time, and both copy a voice from a few seconds of reference audio.

**Continuous latents (Pocket TTS).** Pocket TTS keeps the backbone but drops the tokens {cite}`rouard2025calm`. Instead of choosing codes from a codebook, a small head generates the next frame of a continuous audio latent directly. There is no quantiser throwing information away and no 32-step depth loop per frame, which goes a long way to explaining how 100M parameters can sound this good on two CPU cores.

### How many bits does a voice need?

Chapter 2 measured the phone line at 64 kbit/s: 8,000 μ-law bytes a second. Mimi is far more frugal. Each of its codes picks one of 2,048 entries (11 bits), 12.5 times a second, so every codebook costs 137.5 bit/s. To see what the codebooks buy, I pushed Chapter 4's 80 LibriSpeech recordings through Mimi keeping only the first 1, 2, 4, 8, 16 or all 32 codebooks, and asked Parakeet-TDT to transcribe the result.

```{code-cell} ipython3
books = [json.loads(line) for line in open("data/librispeech/manifest.jsonl")]


def mimi_wer(n):
    e = t = 0
    for r in books:
        errors, words = word_errors(r["written"], heard[f"mimi/{n}/{r['id']}"]["wide"])
        e, t = e + errors, t + words
    return e / t


original = mimi_wer("original")
rows = [(n, n * 11 * 12.5 / 1000, mimi_wer(n)) for n in [1, 2, 4, 8, 16, 32]]
print(f"original recordings: {original:.1%} WER\n\ncodebooks  kbit/s   WER")
for n, kbps, wer in rows:
    print(f"{n:9d}  {kbps:6.2f}  {wer:5.1%}")

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot([r[1] for r in rows], [100 * r[2] for r in rows], "o-", label="through Mimi")
for n, kbps, wer in rows:
    ax.annotate(str(n), (kbps, 100 * wer), textcoords="offset points", xytext=(4, 5), fontsize=8)
ax.axhline(100 * original, ls="--", c="gray", label="original recording")
ax.axvline(64, ls=":", c="gray")
ax.text(64, 30, "phone line\n64 kbit/s ", ha="right", fontsize=8, color="gray")
ax.set(xscale="log", xlabel="bitrate (kbit/s, log scale)", ylabel="WER (%)",
       title="LibriSpeech through Mimi, by number of codebooks")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

for n in [1, 2, 8, 32]:
    print(f"{n} codebook{'s' if n > 1 else ''}:")
    play(f"mimi_{n}")
```

One codebook keeps the rhythm of speech and little else: 42.6% WER, and listening to it is like hearing someone through a wall. Two codebooks are already intelligible. Eight codebooks, 1.1 kbit/s, come within 1.5 points of the original, and all 32 match it at 4.4 kbit/s, about one fifteenth of a phone call. This is why codec language models are practical at all: a second of speech is 12.5 frames, which is a short sequence for a transformer. It is also why their audio can sound good and still say the wrong thing. Mimi reproduces whatever codes it is given, faithfully.

## Say it the way a receptionist would

The fix for the opening is not a better model. It is text that says exactly what should be heard. Production TTS systems call this *text normalisation*, and most of it is boring, specific rules. Here is Halfwheel's, as a function, `speakable()`:

- Tickets are letters, then digits one at a time with "oh" for zero.
- Phone numbers are digits in groups, with a pause (a comma) between groups.
- Prices are pounds and pence in words, the way the shop says them.
- Days, dates, times and ranges are written out in full.
- A small lexicon covers the shop's own words: "Di2" is "D I two", and Shimano's "105" groupset is "one oh five".

```{code-cell} ipython3
import re

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
DIGIT = dict(zip("0123456789", ["oh"] + ONES[1:10]))
DAYS = dict(zip("Mon Tue Wed Thu Fri Sat Sun".split(),
                "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()))
MONTHS = dict(zip("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(),
                  "January February March April May June July August September October November December".split()))
LEXICON = {"Di2": "D I two", "105": "one oh five", "VAT": "V A T"}  # said the way the shop says them


def say(n):
    """A whole number in words, British style: 149 is 'one hundred and forty nine'."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + say(n % 100))
    big, unit = (1000, " thousand") if n < 10**6 else (10**6, " million")
    rest = n % big
    return say(n // big) + unit + ("" if rest == 0 else (" and " if rest < 100 else " ") + say(rest))


def ordinal(n):
    head, _, last = say(n).rpartition(" ")
    odd = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
           "nine": "ninth", "twelve": "twelfth"}
    last = odd.get(last) or (last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return f"{head} {last}".strip()


def clock(h, m="", ampm=""):
    """10:30 is 'ten thirty', 9:05 is 'nine oh five', 2pm is 'two P M'."""
    words = say(int(h))
    if m and m != "00":
        words += " " + ("oh " + ONES[int(m)] if m[0] == "0" else say(int(m)))
    return words + (" " + " ".join(ampm.upper()) if ampm else "")


def speakable(text):
    """Rewrite what a language model writes into what a receptionist would say."""
    day, month = "|".join(DAYS), "|".join(MONTHS)
    t = re.sub(r"\b(" + "|".join(map(re.escape, LEXICON)) + r")\b", lambda m: LEXICON[m[1]], text)
    t = re.sub(r"\bHW-?(\d+)\b", lambda m: "H W " + " ".join(DIGIT[d] for d in m[1]), t)
    t = re.sub(r"\b(\d{3})-(\d{3})-(\d{4})\b",
               lambda m: ", ".join(" ".join(DIGIT[d] for d in g) for g in m.groups()), t)
    t = re.sub(r"\b[A-Z](?:-[A-Z]){2,}\b", lambda m: ", ".join(m[0].split("-")), t)
    t = re.sub(r"£(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d\d))?", lambda m: say(int(m[1].replace(",", ""))) + " pound"
               + ("" if m[1] == "1" else "s") + (" " + say(int(m[2])) if m[2] and m[2] != "00" else ""), t)
    t = re.sub(rf"\b({day})\s+(\d{{1,2}})\s+({month})\b",
               lambda m: f"{DAYS[m[1]]} the {ordinal(int(m[2]))} of {MONTHS[m[3]]}", t)
    t = re.sub(rf"\b({day})-({day})\b", lambda m: f"{DAYS[m[1]]} to {DAYS[m[2]]}", t)
    when = r"(\d{1,2})(?::(\d\d))?\s?(am|pm)"
    t = re.sub(rf"\b{when}-{when}", lambda m: clock(*m.groups()[:3]) + " to " + clock(*m.groups()[3:]), t)
    t = re.sub(rf"\b{when}\b", lambda m: clock(*m.groups()), t)
    t = re.sub(r"\b(\d{1,2}):(\d\d)\b", lambda m: clock(m[1], m[2]), t)
    t = re.sub(r"\b(\d+)-(\d+)\b", lambda m: f"{say(int(m[1]))} to {say(int(m[2]))}", t)
    t = re.sub(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\b", lambda m: say(int(m[0].replace(",", ""))), t)
    return t


for lid, text in LINES:
    if speakable(text) != text:
        print(f"{lid}  {speakable(text)}")
```

Every rewrite is something a person at the counter would say without thinking, which is the test I used for each rule. The order matters: the lexicon runs before the number rules, or "105" would become "one hundred and five".

To check what a listener actually hears, I scored each clip with a *round trip*: Parakeet-TDT transcribes the audio, and I check the transcript against the script. Two scores per clip:

- **Facts:** the 22 things a caller would act on, such as the ticket number, both prices, the date, the time, the phone number and the spelling of Okafor. A fact counts only if it is in the transcript.
- **Word error rate** over the whole line, as in Chapter 4.

Parakeet writes "£149.99" one time and "one hundred and forty nine pounds ninety nine" the next, so both sides go through the same canonical form first: `speakable()`, then Chapter 4's `normalize()`, with spoken hundreds and thousands folded into digits. The models that sample (Pocket TTS, Kyutai and CSM) read the rewritten script three times with different random seeds; Kokoro gives the same audio every time, so it reads it once.

```{code-cell} ipython3
FACTS = {  # what a caller would act on, line by line; a tuple lists equally good ways of hearing it
    "r01": ["Halfwheel Cycles"], "r02": ["HW-1042", "2pm"],
    "r03": ["£25", ("£149.99", "149 pounds and 99 pence")], "r04": ["Thu 14 Nov", "10:30"],
    "r05": ["Priya", "415-555-0192", "6pm"], "r06": ["Okafor", "O-K-A-F-O-R"],
    "r07": ["Shimano 105", "Di2", "3-5 working days"], "r08": ["Mon-Sat", "9am-6pm"],
    "r09": [("£37.50", "37 50", "37 pounds and 50 pence"), "VAT"], "r10": ["Friday", "Brompton"],
    "r11": ["1,500 miles"], "r12": [],
}
SPELLING = {"half wheel": "halfwheel", "grizzle": "grizl"}  # how Parakeet spells two of the shop's names


def fold(s):
    """'1 hundred and 4 9' is 149, '1 thousand 5 0 0' is 1500."""
    s = re.sub(r"\b(\d) hundred(?: and)? (\d) (\d)\b", r"\1 \2 \3", s)
    s = re.sub(r"\b(\d) hundred(?: and)? (\d)\b", r"\1 0 \2", s)
    s = re.sub(r"\b(\d) hundred\b", r"\1 0 0", s)
    s = re.sub(r"\b(\d) thousand (\d) (\d) (\d)\b", r"\1 \2 \3 \4", s)
    return re.sub(r"\b(\d) thousand\b", r"\1 0 0 0", s)


def canon(s):
    """One spoken form for the script and for whatever Parakeet wrote."""
    s = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: ordinal(int(m[1])), s)
    for written, spoken in SPELLING.items():
        s = re.sub(written, spoken, s, flags=re.I)
    s = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", speakable(s), flags=re.I)  # 105DI2 is 105 DI 2
    s = re.sub(r"\b([ap])\.?\s?m\b\.?", r"\1 m", s, flags=re.I)  # 9am, 9 AM and 9 a.m. alike
    return fold(normalize(s))


def score(key, band="wide"):
    """(word errors, reference words, facts heard, facts) for one clip."""
    line = synth[key]["line"]
    ref, hyp = canon(dict(LINES)[line]).split(), canon(heard[key][band]).split()
    padded = f" {' '.join(hyp)} "
    facts = [any(f" {canon(f)} " in padded for f in (alts if isinstance(alts, tuple) else (alts,)))
             for alts in FACTS[line]]
    return sum(edit_ops(ref, hyp).values()), len(ref), sum(facts), len(facts)


def totals(model, variant, band="wide"):
    s = np.array([score(k, band) for k, r in synth.items() if r["model"] == model and r["variant"] == variant])
    return f"{s[:, 2].sum():2d}/{s[:, 3].sum()} facts, {s[:, 0].sum() / s[:, 1].sum():5.1%} WER"


print(f"{'':16s} {'as written':>24s}   {'speakable()':>24s}")
for m, name in MODELS.items():
    print(f"{name:16s} {totals(m, 'raw'):>24s}   {totals(m, 'spoken'):>24s}")
```

Normalising fixed most of the errors for every model, and nearly all of them for two. Read as written, the models got 11 to 15 of 22 facts right. Through `speakable()`, Pocket TTS and Kyutai each got 65 of 66 over three reads, with word error rates under 1%. Here is the opening line again, rewritten:

```{code-cell} ipython3
print(f"{'speakable()':16s} {speakable(dict(LINES)['r05'])}")
for m, name in MODELS.items():
    print(f"{name:16s} {heard[f'{m}/r05_spoken_0']['wide']}")
    play(f"{m}_r05_spoken")
```

The errors that are left are interesting. Pocket's one miss was "Shimano I O five", a slip on a single read. Kyutai's was "Okefor". Kokoro's two misses are both on Okafor. One is real: its front end says `ˈɒkafə`, "OCK-a-fuh", for a name pronounced "oh-KAH-for". The other is the judge's fault, not Kokoro's. The phonemes show it spelling O, K, A, F, O, R correctly, but Parakeet ran the letters together on the wideband audio, and heard them correctly over the phone line. Pronouncing names is where a lexicon, or a model you can direct, earns its place, and it is the subject of Chapter 7.

CSM-1B is the outlier, with 54 of 66 facts even after normalising. I come back to why below.

## Measuring a voice

A receptionist's voice has to do three things: say the right thing, start quickly, and survive the phone. The round trip covers the first, though with two blind spots:

- **The recogniser hides some errors.** Kokoro said "ten forty two" for the ticket, and Parakeet wrote "HW1042", exactly what it writes for "one oh four two". The words were wrong; the transcript was right. Phonemes, or your ears, catch what a round trip can't.
- **The recogniser makes its own mistakes**, like the run-together spelling above. Treat a single miss as a reason to listen, not a verdict.

### How soon does it start talking?

Chapter 5 spent a whole chapter shaving tenths of a second off listening. Speaking has its own delay: the *time to first audio*, from handing over the text to the first sound coming out. A model that streams can start playing after its first chunk; one that doesn't makes the caller wait for the whole sentence. The other number is speed: how many seconds of audio it makes per second of work. Below 1, it can't keep up with itself.

```{code-cell} ipython3
speed = []
for m, name in MODELS.items():
    for device, where in [("gpu", "L4 GPU"), ("cpu", "CPU")]:
        runs = [r for r in synth.values() if r["model"] == m and device in r]
        if runs:
            label = where if device == "gpu" else f"CPU, {runs[0]['cpu']['threads']} threads"
            first = np.median([r[device]["ttfa"] for r in runs])
            rate = np.median([r["seconds"] / r[device]["wall"] for r in runs])
            speed.append((f"{name}, {label}", first, rate))

print(f"{'':28s} {'first audio':>11s} {'x real time':>12s}")
for label, first, rate in speed:
    print(f"{label:28s} {first:10.2f}s {rate:11.1f}x")

fig, ax = plt.subplots(figsize=(7, 2.8))
ax.barh([s[0] for s in speed][::-1], [s[1] for s in speed][::-1], color="#4a7fb0")
for i, (label, first, rate) in enumerate(speed[::-1]):
    ax.text(first * 1.1, i, f"{first:.2f} s, {rate:.1f}x real time", va="center", fontsize=8)
ax.set(xscale="log", xlim=(0.05, 400), xlabel="median time to first audio (s, log scale)")
plt.tight_layout()
plt.show()
```

Four very different deals:

- **Kokoro on a GPU** is effectively instant: 0.09 s to the first audio, 60 times faster than real time. On a CPU it is still 2.6 times real time, but it renders a whole sentence before you hear any of it, so the caller waits 1.7 s.
- **Pocket TTS** streams, so on two CPU threads it started talking in 0.19 s, which matches Kyutai's own figure. For a receptionist without a GPU, that is the number to beat.
- **Kyutai TTS 1.6B** took 0.96 s on the L4 through Kyutai's PyTorch script. Part of that is the 16-frame head start the text stream gets; the rest is a Python loop running one frame at a time. Kyutai reports 150 ms from its own optimised server {cite}`zeghidour2025dsm`. Measure the serving stack you will actually run, not the paper.
- **CSM-1B through Hugging Face transformers** doesn't stream: `generate()` returns the whole clip, which took 11 s for a median line and ran at half real time. It can't hold a phone conversation run this way.

### Does it say the same thing every time?

Kokoro is deterministic: same text, same audio. The other three sample, so every read is a fresh roll of the dice. For each model, I counted the lines it got completely right (every fact, and at most one word different) on all three reads, and the clips that ran away: they kept generating up to my 20 s cap on a line that takes about 5 s to say.

```{code-cell} ipython3
SEEDS = [0, 1, 2]
spoken_len = {r["line"]: r["seconds"] for r in synth.values() if r["model"] == "kokoro" and r["variant"] == "spoken"}
for m in ["pocket", "kyutai", "csm"]:
    clean = sum(all(score(f"{m}/{lid}_spoken_{s}")[2] == score(f"{m}/{lid}_spoken_{s}")[3]
                    and score(f"{m}/{lid}_spoken_{s}")[0] <= 1 for s in SEEDS) for lid, _ in LINES)
    clips = [r for r in synth.values() if r["model"] == m]
    runaway = sum(bool(r.get("hit_cap")) for r in clips)
    longest = max(r["seconds"] / spoken_len[r["line"]] for r in clips)
    print(f"{MODELS[m]:16s} lines right on all 3 reads: {clean:2d}/12   ran to the cap: {runaway}/{len(clips)}   "
          f"longest clip: {longest:.1f}x Kokoro's length")

print("\nCSM reading r12, 20 s for a 4 s line. Parakeet heard:", heard["csm/r12_spoken_0"]["wide"])
play("csm_r12_runaway")
```

Pocket TTS and Kyutai read 10 and 11 of the 12 lines right on every read, never ran away, and never took more than twice as long as Kokoro to say a line. CSM-1B read 3 of 12 right every time, and 4 of its 48 clips ran to the cap.

CSM-1B's failures all have the same shape: it is a *conversational* model, trained to continue a dialogue, and it treats the end of my text as a suggestion. On r12 ("Mm, let me just check that for you. Okay, yes, I can see it here.") it said the first sentence and then kept going for 20 s, in two of three reads. It dropped digits from the phone number in two of three reads, and spelled Okafor as "O K A A F O". None of this means CSM is a bad model. It means its job, sounding natural in context, isn't this job: reading a fixed sentence exactly. If you use a sampled model for fixed text, cap the length relative to the text, check the audio's duration before you play it, and be ready to retry.

### Does it survive the phone line?

Everything above was scored on the clean 24 kHz output. The caller hears it through Chapter 2's phone line: 8 kHz, 300 to 3,400 Hz, line noise at 25 dB below the speech, then μ-law.

```{code-cell} ipython3
print(f"{'speakable()':16s} {'wideband':>24s}   {'phone line':>24s}")
for m, name in MODELS.items():
    print(f"{name:16s} {totals(m, 'spoken'):>24s}   {totals(m, 'spoken', 'phone'):>24s}")
```

It does. No model lost a fact over the phone line; each gained one, which is within the recogniser's own noise, and Pocket TTS and Kyutai got all 66. That matches what Chapter 4 found for Halfwheel's synthetic callers: clean synthetic speech has none of the room noise, distance or mumbling that makes real phone audio hard, so band-limiting it costs little. The phone line is not where your TTS budget goes.

### Licences, briefly

All four models are open: Kokoro and CSM-1B under Apache 2.0, Kyutai TTS 1.6B and Pocket TTS under CC BY 4.0. The voices are a separate question. VCTK, the speaker I cloned here, is CC BY 4.0, while some of Kyutai's voice library (the Expresso voices) is non-commercial only. Check the voice as well as the model, and only clone a voice with its owner's consent.

## Halfwheel's voice

Putting the chapter together, the receptionist speaks like this:

1. The language model writes the reply as text, the way it writes everything.
2. `speakable()` rewrites it for the ear: tickets, phone numbers, prices, dates, times and the shop's lexicon.
3. Pocket TTS reads it on the CPU, with the first audio out in about 0.2 s. With a GPU to spare, Kokoro is the alternative: instant, deterministic, and easy to debug through its phonemes, as long as the whole sentence fits in the wait.
4. Before a sampled model's clip is played, check that its length is plausible for the text (a few words a second), and retry if not.

That puts about 0.2 s of speaking delay on top of Chapter 5's 1.3 s of listening. Chapter 8 joins the two halves into a receptionist that answers the phone.

## When it breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| Numbers read as amounts ("ten forty two" for a ticket, "four hundred fifteen" for an area code) | The model is guessing how to say digits | Rewrite them as words before TTS, the way your business says them |
| "Mon-Sat" or "Di2" comes out as a strange word | Abbreviations and product names the model has never seen | A lexicon of your own words, applied before the number rules |
| A name is mispronounced every time | The phoneme front end guessed from the spelling | A respelling or phoneme override for that name (Chapter 7) |
| A clip runs on long after the sentence, or stops halfway | A sampled model treating the text as a prompt | Cap generation length relative to the text, check the duration, retry |
| A long silence before the receptionist speaks | A model that renders the whole utterance before playing | Stream, or split the reply into sentences and start with the first |
| The round trip says a fact was wrong but it sounds right | The recogniser's own error | Listen; for spellings and codes, check the model's phonemes |

```{tip}
**Takeaways**
- Text written for a screen is not text written for the ear. Normalise tickets, numbers, prices, dates and your own vocabulary before any TTS model sees them; that one function did more than switching models.
- Open TTS comes in three designs: phonemes and a vocoder (fast, deterministic, debuggable), audio tokens from a codec language model (expressive, sampled), and continuous latents (small enough for a CPU).
- A codec makes speech cheap to model: 12.5 frames a second, and 1.1 kbit/s is enough to keep the words.
- Measure time to first audio on the stack you will serve, not the one in the paper. Streaming matters as much as raw speed.
- Sampled models need guard rails: cap the length, check the duration, retry.
- A round trip through ASR is a good, cheap check. It hides some errors and invents others, so listen to every miss.
```

## Try it yourself

The companion notebook, `chapters/06-how-tts-works`, has the saved outputs, every cell above, and the capture script.

1. **Your own script.** Replace `LINES` with ten things your own voice agent says. Which of `speakable()`'s rules do you need, and which new ones?
2. **Break the order.** Move the lexicon line in `speakable()` to the end. What happens to "Shimano 105"?
3. **Fix Okafor.** Kokoro accepts phonemes inline, as in `[Okafor](/Qkˈɑːfɔː/)` (in Kokoro's British symbols, `Q` is the "oh" sound). Rerun r06 through Kokoro with and without it, and compare the phonemes.
4. **Fewer codebooks.** Kyutai's model generates all 32 of Mimi's codebooks. Change `n_q=32` to `n_q=8` in `kyutai()` in `sim/capture_ch6.py` and rerun. How much faster is it, and what happens to the facts?

## Next

The receptionist now says the right words. It says them all the same way, though: the same pace, the same pitch, the same cheerful flatness for "your bike is ready" and "I'm sorry, we've lost your wheel". Chapter 7 takes the models that can be directed and makes the receptionist sound like it means what it says.
