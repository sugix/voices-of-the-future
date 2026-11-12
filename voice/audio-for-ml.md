---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
date: "2026-11-30"  # quoted: myst-nb stores front matter as JSON
description: What a phone call looks like to a speech model, built from scratch on one Halfwheel call, and what common audio mistakes really cost.
---

# Audio for ML engineers

A customer rings Halfwheel Cycles and says, "Hi, is my bike ready? The ticket number is HW-1042." That is five seconds of speech. By the time it reaches the receptionist, it is 40,000 numbers, each squeezed into a single byte, and a good part of what made the voice sound like a person in the room is gone.

Every chapter after this one hands audio like that to a model. In my experience most voice bugs are not model bugs. They are audio bugs: the wrong sample rate, a "resampler" that was really just slicing, a voice detector that answered the hold music. This chapter is the minimum audio you need to stop making them, built from scratch on that one call and then measured on all thirty of Halfwheel's test calls.

Here is what I found:

| Finding | Number |
|---|---|
| A phone call is small | 64 kbit/s, six times less data than the clean recording |
| The phone line drops everything above 3.4 kHz | only 7% of the energy, but most of it in 25 of 173 frames: the *s*, *f* and *ch* sounds |
| Downsampling by keeping every third sample invents sounds | a 5 kHz tone reappears at 3 kHz at full strength; a proper resampler leaves it at -57 dB |
| μ-law is why 8 bits are enough | quiet speech keeps 34 dB of signal-to-noise, against 9 dB for plain 8-bit audio |
| Feeding 8 kHz audio to a 16 kHz model without resampling | word error rate from 5% to 46%, and only 10 of 48 tickets, names and bikes heard, down from 30 |
| Two mistakes the model shrugged off, and one it did not | aliasing and unscaled samples cost almost nothing; clipping nearly doubled errors on real speech |
| A loudness threshold cannot find speech | any setting that hears 94% of speech also answers the hold music; Silero VAD hears 97% and ignores it |

```{note}
Everything in this chapter runs on a CPU in seconds. The companion notebook is `chapters/02-audio-for-ml` in [github.com/sugix/voices-of-the-future](https://github.com/sugix/voices-of-the-future). The one model experiment uses saved transcripts, and the notebook shows how to rerun it on a free Colab GPU or, more slowly, on your laptop.
```

## Sound is a list of numbers

A microphone measures air pressure many thousands of times a second. Each measurement is a **sample**, the number of samples per second is the **sample rate**, and the number of bits used to store each one is the **bit depth**. A WAV file is those samples with a short header saying how to read them. The header is simple enough to parse by hand, and doing it once removes a lot of mystery:

```{code-cell} ipython3
import json
import struct
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display
from scipy import signal


def read_wav(path):
    """Parse a WAV file by hand: a few labelled chunks of header, then the raw sample bytes."""
    data = Path(path).read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", "not a WAV file"
    pos, info = 12, {"bytes": len(data)}
    while pos + 8 <= len(data):
        chunk, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if chunk == b"fmt ":
            code, channels, sr, byte_rate, _, bits = struct.unpack("<HHIIHH", body[:16])
            info.update(code=code, channels=channels, sr=sr, byte_rate=byte_rate, bits=bits)
        elif chunk == b"data":
            info["raw"] = body
        pos += 8 + size + (size & 1)  # chunks are padded to an even length
    return info


for path in ["data/ch2/c01_status_24k.wav", "data/calls/phone/c01_status.wav"]:
    w = read_wav(path)
    print(f"{path:34} format {w['code']}, {w['channels']} channel, {w['sr']:,} Hz, {w['bits']} bits, "
          f"{w['byte_rate'] * 8 / 1000:.0f} kbit/s, {len(w['raw']) / w['byte_rate']:.2f} s, {w['bytes']:,} bytes")
```
The clean version is what the speech synthesiser produced: 24,000 samples a second, 16 bits each. The phone version is what a phone network delivers: 8,000 samples a second, 8 bits each. That is 64 kbit/s, the rate of G.711, the encoding digital phone networks have carried since the 1970s and that most cloud telephony still offers today. Format 1 means plain PCM, where each sample is a number. Format 7 means **μ-law**, where each byte is a compressed number that needs decoding:

```{code-cell} ipython3
def mulaw_decode(raw):
    """G.711 mu-law: each byte is a sign bit, a 3-bit exponent and a 4-bit mantissa, stored inverted."""
    u = ~np.frombuffer(raw, dtype=np.uint8)
    exponent, mantissa = (u >> 4) & 7, (u & 15).astype(np.int32)
    magnitude = (((mantissa << 3) + 0x84) << exponent) - 0x84
    return np.where(u & 0x80, -magnitude, magnitude) / 32768


def load(path):
    """Samples as float32 in [-1, 1], and the sample rate."""
    w = read_wav(path)
    if w["code"] == 1 and w["bits"] == 16:
        x = np.frombuffer(w["raw"], dtype="<i2") / 32768
    elif w["code"] == 7:
        x = mulaw_decode(w["raw"])
    else:
        raise ValueError(f"WAV format {w['code']} is not handled here")
    if w["channels"] > 1:
        x = x.reshape(-1, w["channels"]).mean(axis=1)
    return x.astype(np.float32), w["sr"]


clean, clean_sr = load("data/ch2/c01_status_24k.wav")
phone, phone_sr = load("data/calls/phone/c01_status.wav")
i = int(1.3 * phone_sr)  # 1.3 seconds in, mid-word
raw = read_wav("data/calls/phone/c01_status.wav")["raw"]
print("phone bytes  ", list(raw[i:i + 6]))
print("as samples   ", phone[i:i + 6].round(4))
print(f"clean: {len(clean):,} samples at {clean_sr:,} Hz; phone: {len(phone):,} samples at {phone_sr:,} Hz")
display(Audio(clean, rate=clean_sr), Audio(phone, rate=phone_sr))
```
Those ten lines of decoding match the `soundfile` library on all thirty phone files, bit for bit. Play both versions. The phone one is unmistakably a phone call, but every word is still there.

Plotted, the two look alike from a distance. Up close, the clean recording has three samples for every one of the phone version, and the phone version has lost the fine wiggles:

```{code-cell} ipython3
fig, ax = plt.subplots(3, 1, figsize=(9, 6))
for a, x, sr, name, c in [(ax[0], clean, clean_sr, "clean, 24 kHz", "C0"), (ax[1], phone, phone_sr, "phone, 8 kHz", "C1")]:
    t = np.arange(len(x)) / sr
    a.plot(t, x, lw=0.4, color=c)
    a.set(ylabel="amplitude", title=name)
    zoom = (t >= 0.42) & (t < 0.44)
    ax[2].plot(t[zoom] * 1000, x[zoom] / np.abs(x).max(), ".-", ms=3, lw=0.6, color=c, label=name)
ax[1].set_xlabel("seconds")
ax[2].set(xlabel="milliseconds", ylabel="scaled amplitude", title="20 ms of the first word: every dot is one sample")
ax[2].legend(loc="upper right", fontsize=8)
fig.tight_layout()
```
## What the phone line takes away

### Everything above 3.4 kHz

A sample rate can only represent frequencies up to half its value, the **Nyquist frequency**. At 8 kHz that is 4 kHz, and phone networks filter further, keeping roughly 300 to 3,400 Hz. Here is where the energy of Halfwheel's call sits:

```{code-cell} ipython3
f, p = signal.welch(clean, clean_sr, nperseg=1024)
above = p[f > 3400].sum() / p.sum()
frames = clean[: len(clean) // 480 * 480].reshape(-1, 480)  # 20 ms frames
spec = np.abs(np.fft.rfft(frames * np.hanning(480), axis=1)) ** 2
fb = np.fft.rfftfreq(480, 1 / clean_sr)
loud = (frames**2).mean(axis=1) > 1e-5
hi_share = spec[:, fb > 3400].sum(axis=1) / spec.sum(axis=1)
print(f"energy above 3.4 kHz: {above:.1%} of the call")
print(f"20 ms frames with sound: {loud.sum()}, of which {np.sum(loud & (hi_share > 0.5))} have most energy above 3.4 kHz")

fp, pp = signal.welch(phone, phone_sr, nperseg=512)
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.semilogy(f, p, lw=1, label="clean, 24 kHz")
ax.semilogy(fp, pp * (p.max() / pp.max()), lw=1, label="phone, 8 kHz (scaled)")
ax.axvspan(300, 3400, color="C2", alpha=0.1, label="phone band, 300 to 3,400 Hz")
ax.set(xlabel="frequency (Hz)", ylabel="power", title="Where the sound energy is", xlim=(0, 12000))
ax.legend(fontsize=8)
fig.tight_layout()
```
Only 7% of the call's energy lies above 3.4 kHz, which is why the phone version still sounds intelligible. But that 7% is not spread evenly. In 25 of the 173 frames that contain sound, *most* of the energy is up there. Those are the hissing and rushing consonants: *s*, *f*, *ch*, *th*. They carry little energy and a lot of information, and they are exactly what the phone line removes. It is why people spell names over the phone with "S for Sugar, F for Freddie".

It costs speech models too. Chapter 4's test bench includes 80 real recordings from LibriSpeech, and over the phone line every model made between 1.4 and 2.2 times as many word errors as on the clean versions. On Halfwheel's synthesised calls it barely mattered, because synthesised speech is unnaturally crisp. Keep that in mind whenever you test with generated audio: it flatters every model.

### Resampling: filter first, or you invent sounds

You will resample all the time: 8 kHz phone audio up to the 16 kHz most recognisers expect, 24 kHz synthesised speech down to 8 kHz for the phone line. The tempting shortcut for going down is to keep every third sample. Here is a tone sweeping up from 100 Hz to almost 12 kHz, downsampled to 8 kHz both ways:

```{code-cell} ipython3
sr = 24000
t = np.arange(2 * sr) / sr
chirp = 0.5 * signal.chirp(t, f0=100, t1=2, f1=11900)  # a tone sweeping up to 11.9 kHz
sliced = chirp[::3]  # 'downsample' to 8 kHz by keeping every third sample
proper = signal.resample_poly(chirp, 1, 3)  # filter first, then keep every third sample

tone = np.sin(2 * np.pi * 5000 * t)  # a 5 kHz tone cannot exist at 8 kHz
for name, x in [("every third sample", tone[::3]), ("resample_poly", signal.resample_poly(tone, 1, 3))]:
    mid = x[1000:-1000]  # skip the filter's start-up at each end
    spec = np.abs(np.fft.rfft(mid * np.hanning(len(mid))))
    freqs = np.fft.rfftfreq(len(mid), 1 / 8000)
    level = 20 * np.log10(np.sqrt(np.mean(mid**2)) / np.sqrt(0.5))  # relative to the original tone
    print(f"{name:20} strongest frequency {freqs[spec.argmax()]:5.0f} Hz, level {level:6.1f} dB")

fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
for a, x, name in [(ax[0], sliced, "every third sample"), (ax[1], proper, "resample_poly")]:
    a.specgram(x, Fs=8000, NFFT=256, noverlap=192, cmap="magma", vmin=-120)
    a.set(title=name, xlabel="seconds")
ax[0].set_ylabel("frequency (Hz)")
fig.tight_layout()
display(Audio(sliced, rate=8000), Audio(proper, rate=8000))
```
Keeping every third sample **folds** every frequency above 4 kHz back down into the band: the sweep hits 4 kHz, then turns around and comes back down as a sound that was never there. A 5 kHz tone reappears at 3 kHz (8,000 minus 5,000) at full strength. `resample_poly` low-pass filters before it drops samples, so the same tone ends up 57 dB quieter, which is effectively gone. This folding is called **aliasing**, and in speech it smears the high hiss of an *s* down into the middle of the band, where you can hear it as a harsh, lispy edge. (Later in this chapter you will see that a speech model minds this much less than your ear does. Use a real resampler anyway.) Use a real resampler (`scipy.signal.resample_poly`, `torchaudio.functional.resample`, `librosa.resample`) every time, and be suspicious of any code that indexes with a step.

### μ-law: 8 bits that behave like 14

Eight bits give 256 levels. Spaced evenly, quiet speech only uses a handful of them, and the rounding error becomes nearly as loud as the speech. μ-law spaces the levels logarithmically, so quiet sounds get fine steps and loud sounds get coarse ones. Here is the signal-to-noise ratio of each scheme as the caller gets quieter:

```{code-cell} ipython3
MU = 255


def mulaw_8bit(x):
    """Compress, round to 8 bits, expand: the continuous form of G.711."""
    y = np.sign(x) * np.log1p(MU * np.abs(x)) / np.log1p(MU)
    y = np.round(y * 127) / 127
    return np.sign(y) * np.expm1(np.abs(y) * np.log1p(MU)) / MU


def linear_8bit(x):
    return np.round(x * 127) / 127


def snr_db(x, y):
    return 10 * np.log10(np.sum(x**2) / np.sum((x - y) ** 2))


speech = signal.resample_poly(clean, 1, 3)
speech = speech / np.abs(speech).max()
print("peak level   linear 8-bit   mu-law 8-bit")
for level in [0, -10, -20, -30, -40]:
    x = speech * 10 ** (level / 20)
    print(f"{level:6} dB   {snr_db(x, linear_8bit(x)):9.1f} dB   {snr_db(x, mulaw_8bit(x)):9.1f} dB")
```
At full volume the two are equal. At -30 dB, a soft-spoken caller on a cheap headset, plain 8-bit audio keeps 9 dB of signal over the rounding noise, which sounds like a radio between stations, while μ-law keeps 34 dB. That is why the phone network uses it, and why you must decode it rather than read the bytes as numbers.

### The whole phone line in one function

Put the pieces together and you have the function that made every phone file in this book. It lives in `sim/halfwheel_calls.py`, and here it is in full:

```{code-cell} ipython3
PHONE_SR = 8_000


def to_phone(audio, sr, snr_db, rng):
    """Simulate a phone line: 8 kHz, 300-3400 Hz band, background noise."""
    down = signal.resample_poly(audio, PHONE_SR, sr)
    sos = signal.butter(4, [300, 3400], btype="bandpass", fs=PHONE_SR, output="sos")
    band = signal.sosfilt(sos, down)
    noise = rng.normal(0.0, 1.0, size=band.shape)
    p_sig = np.mean(band**2)
    noise *= np.sqrt(p_sig / (10 ** (snr_db / 10)) / np.mean(noise**2))
    out = band + noise
    return (0.9 * out / np.max(np.abs(out))).astype(np.float32)


mine = to_phone(clean, clean_sr, snr_db=25, rng=np.random.default_rng(0))
for name, x in [("my to_phone", mine), ("the book's phone file", phone)]:
    f2, p2 = signal.welch(x, PHONE_SR, nperseg=512)
    band = (f2 >= 300) & (f2 <= 3400)
    print(f"{name:22} {len(x):6} samples, {p2[band].sum() / p2.sum():.1%} of power inside 300-3,400 Hz")
```
My rebuild and the book's file keep the same share of their power inside the phone band. The noise differs because the random seed does, which is the point of seeding: rerun the book's generator and you get the book's files exactly. The μ-law step happens when the file is written (`soundfile.write(..., subtype="ULAW")`), which is the step you just decoded by hand.

## What a speech model actually sees

Almost no speech recogniser reads samples directly. It reads a **log-mel spectrogram**: the audio cut into overlapping 25 ms windows every 10 ms, each window turned into a spectrum with a Fourier transform, the spectrum pooled into 80 bands spaced the way human hearing is (fine at low frequencies, coarse at high ones), and the result put on a log scale. That is all the feature extraction there is, and it fits in one function:

```{code-cell} ipython3
def log_mel(x, sr=16000, n_fft=512, win=400, hop=160, n_mels=80):
    """The features most speech models read: 25 ms windows every 10 ms, 80 mel bands, log power."""
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop] * np.hanning(win)
    power = np.abs(np.fft.rfft(frames, n_fft)) ** 2
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = hz(np.linspace(mel(0), mel(sr / 2), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    weights = np.maximum(0, np.minimum((bins - lo) / (mid - lo), (hi - bins) / (hi - mid)))
    return np.log(power @ weights.T + 1e-10), edges[1:-1]


clean16 = signal.resample_poly(clean, 2, 3)
phone16 = signal.resample_poly(phone, 2, 1)
M_clean, centres = log_mel(clean16)
M_phone, _ = log_mel(phone16)
top = centres > 3400
print(f"{M_clean.shape[0]} frames x {M_clean.shape[1]} bands; {top.sum()} bands sit above 3.4 kHz")
ratio = np.exp(M_phone[:, top]).sum() / np.exp(M_clean[:, top]).sum()
print(f"energy in those bands, phone against clean: {10 * np.log10(ratio):.0f} dB")
print(f"bands above 4 kHz: {np.sum(centres > 4000)}, and the phone version cannot fill any of them")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for a, M, name in [(ax[0], M_clean, "clean"), (ax[1], M_phone, "phone")]:
    a.imshow(M.T, origin="lower", aspect="auto", cmap="magma", vmin=M_clean.max() - 18, vmax=M_clean.max(),
             extent=[0, M.shape[0] / 100, 0, 80])
    a.set(title=f"log-mel, {name}", xlabel="seconds")
ax[0].set_ylabel("mel band")
fig.tight_layout()
```
That is 100 frames a second, each a column of 80 numbers; Parakeet uses 128 bands, Whisper large-v3 also 128, and the idea is identical. On the clean call, the top quarter of the picture is full of consonants. On the phone call, the 19 bands above 4 kHz are empty, because an 8 kHz signal cannot contain anything there however carefully it is upsampled, and the bands between 3.4 and 4 kHz are dimmed. A model trained on full-band speech sees a quarter of its input go dark on every phone call. It copes, mostly, but this is the gap that phone-specific training data, and Chapter 4's tricks, are there to close.

These frames are the starting point for Chapter 3. Parakeet's encoder compresses every eight of them into one 80 ms frame for its decoder to read. Some newer models, such as Kyutai's and the Moshi family in Chapter 10, skip the spectrogram and learn their own audio codec from raw samples, but they still care just as much about sample rate.

## Six ways to hand audio to a model, measured

Every mistake in this chapter is easy to make in real code, so I measured what each one costs. The capture script (`sim/capture_ch2.py`) fed Halfwheel's 30 calls and the 80 LibriSpeech recordings to Parakeet-TDT, the model Halfwheel ends up choosing in Chapter 4, seven different ways. The scoring functions are the ones Chapter 4 builds and explains; here I just import them.

```{code-cell} ipython3
sys.path.insert(0, "sim")
from scoring import entity_hits, word_errors

refs = {f"halfwheel/{r['id']}": r for r in map(json.loads, open("data/calls/manifest.jsonl"))}
refs |= {f"librispeech/{r['id']}": r for r in map(json.loads, open("data/librispeech/manifest.jsonl"))}
cap = json.load(open("data/ch2/capture_ch2.json"))


def score(res, which):
    errs = [word_errors(refs[k]["written"], r["text"]) for k, r in res.items() if k.startswith(which)]
    return sum(e for e, _ in errs) / sum(n for _, n in errs)


print(f"{'what Parakeet-TDT was given':52} {'Halfwheel':>10} {'LibriSpeech':>12} {'entities':>9}")
for cond, res in cap["asr"].items():
    hits = [ok for k, r in res.items() if k.startswith("halfwheel") for _, ok in entity_hits(refs[k], r["text"])]
    print(f"{cond:52} {score(res, 'halfwheel'):10.1%} {score(res, 'librispeech'):12.1%} {sum(hits):5}/{len(hits)}")
print()
for cond, res in cap["asr"].items():
    print(f"{cond[:44]:44} {res['halfwheel/c01_status']['text']}")
```
The rows are worth reading one at a time.

- **The right way.** Clean audio resampled to 16 kHz, and phone audio resampled to 16 kHz. The phone line costs nothing on the synthesised calls and about half as many errors again on real speech (2.3% to 3.6%).
- **Not resampling.** Hand an 8 kHz signal to a model that assumes 16 kHz and every sound plays at double speed and an octave up. "Is my bike ready? The ticket number is" became "is my mouth card, the titanium". Errors rose eight to fifteen times, and the entities heard fell from 30 to 10. No exception is raised; the model happily transcribes chipmunk speech. This is the most common audio bug I see, and the most expensive.
- **Downsampling by slicing.** Audible, as you heard, but the model barely noticed: 2.6% against 2.8% on LibriSpeech. Speech has little energy above 4 kHz to fold back, so there is little to corrupt. Still use a real resampler. It costs nothing, and your next model or your next caller may mind more.
- **Clipping.** Turning the phone audio up eight times and clipping it at full scale left the synthesised calls alone, but nearly doubled the errors on real speech, from 3.6% to 6.6%. Real voices have loud peaks that clip; synthesised ones, normalised to a safe level, mostly did not. It is the same lesson as before: synthesised test audio hides problems.
- **Unscaled int16 samples.** Feeding the raw integers (up to 32,767) instead of floats between -1 and 1 changed nothing, because Parakeet normalises each feature over the utterance (`normalize: per_feature` in its configuration). Anything that relies on absolute levels does break, such as the loudness-based detector in the next section, or any check for clipping. Scale your audio.

The pattern: a model forgives you for small changes to the *signal* and punishes you for mistakes about its *timeline*. Check the sample rate first, every time.

## Is anyone speaking? Voice activity detection

Before the receptionist transcribes anything, it has to know whether anyone is talking. That sounds trivial until a caller is put on hold, or rings from the roadside, or from a workshop. A **voice activity detector** (VAD) answers one question per short frame: speech or not. Chapter 5 builds on it to decide when a caller has finished their turn, and when they are interrupting.

To test one, I need audio where I know the answer. The capture script strings eight Halfwheel calls together and puts three seconds of something else before each one, sent through the same phone line: a quiet open line, a truck passing the caller, the shop's hold music, and a mechanic tapping a wrench in the workshop. The ground truth comes from the clean recordings, so every 32 ms frame is labelled speech, not speech, or (for the short pauses inside a call) not scored.

The simplest VAD is a loudness threshold: estimate the noise floor from the quietest frames, and call anything some margin above it speech. Here is that detector at five margins:

```{code-cell} ipython3
truth = json.load(open("data/ch2/vad_truth.json"))
stream, sr = load("data/ch2/vad_stream.wav")
FRAME = truth["frame"]
labels = np.array(truth["labels"])
frames = stream[: len(labels) * FRAME].reshape(-1, FRAME)
energy_db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-10)
kind = np.empty(len(labels), dtype=object)
for seg in truth["segments"]:
    a = round(seg["start_s"] * sr / FRAME)
    kind[a:a + round(seg["seconds"] * sr / FRAME)] = seg["kind"]


def energy_vad(db, margin=15):
    floor = np.percentile(db, 10)  # the quietest frames are the line noise
    return db > floor + margin


def report(name, speech):
    out = {"VAD": name, "speech heard": np.mean(speech[labels == 1])}
    for k in ["quiet line", "road noise", "hold music", "workshop clatter"]:
        out[k] = np.mean(speech[(labels == 0) & (kind == k)])
    return out


print(f"{'margin':>8} {'speech heard':>13}" + "".join(f"{k:>18}" for k in ["quiet line", "road noise", "hold music", "workshop clatter"]))
for margin in [10, 15, 20, 25, 30]:
    r = report("", energy_vad(energy_db, margin))
    print(f"{margin:5} dB {r['speech heard']:13.0%}" + "".join(f"{v:18.0%}" for k, v in list(r.items())[2:]))
```
There is no good setting. With a margin low enough to hear the speech, the hold music counts as a caller every time, and the passing truck most of the time. Raise the margin until the music goes quiet, and the detector misses more than a third of what people say. Loudness simply is not what makes speech speech.

**Silero VAD** is a small neural network (about 2 MB) trained to tell speech from everything else, and it runs on 8 kHz audio in 32 ms frames, faster than real time on any CPU. The capture script ran it on the same stream:

```python
# pip install silero-vad
import torch
from silero_vad import load_silero_vad

model = load_silero_vad()
probs = [model(torch.from_numpy(stream[i:i + 256].copy()), 8000).item()  # one 32 ms frame at a time
         for i in range(0, len(stream) - 255, 256)]
```

```{code-cell} ipython3
silero = np.array(cap["silero"])
results = [report("energy, 15 dB", energy_vad(energy_db)), report("Silero VAD", silero > 0.5)]
print(f"{'':16}" + "".join(f"{k:>18}" for k in results[0] if k != "VAD"))
for r in results:
    print(f"{r['VAD']:16}" + "".join(f"{v:18.0%}" for k, v in r.items() if k != "VAD"))

t = np.arange(len(labels)) * FRAME / sr
fig, ax = plt.subplots(2, 1, figsize=(9, 4.4), sharex=True)
ax[0].plot(np.arange(len(stream)) / sr, stream, lw=0.3, color="0.4")
for seg in truth["segments"]:
    if seg["kind"] != "speech":
        for a in ax:
            a.axvspan(seg["start_s"], seg["start_s"] + seg["seconds"], color="C3", alpha=0.12)
        ax[0].text(seg["start_s"] + 0.1, 0.8, seg["kind"], fontsize=7)
ax[0].set(ylabel="amplitude", title="Eight calls, and what came down the line between them (shaded)")
ax[1].fill_between(t, 0, energy_vad(energy_db), step="post", alpha=0.5, label="energy, 15 dB")
ax[1].fill_between(t, 0, -(silero > 0.5).astype(float), step="post", alpha=0.5, color="C2", label="Silero")
ax[1].set(yticks=[], xlabel="seconds", title="Where each detector says someone is speaking")
ax[1].legend(loc="lower right", fontsize=8)
fig.tight_layout()
```
Silero hears 97% of the speech, ignores the truck and the workshop entirely, and mistakes 2% of the hold music for a voice. The picture shows the difference at a glance: the energy detector lights up on almost everything loud, while Silero lights up on the calls.

Two caveats. My hold music is a few synthesised chords, and real hold music often has singing in it, which any honest VAD will call speech. And the truck and the wrench were synthesised too. They are a fair test of loudness against speechiness, not a substitute for recordings of your own callers' backgrounds.

## When it breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| Fast, high-pitched gibberish, or a near-empty transcript | Audio at one sample rate, model expecting another | Check the rate in the file header and resample to what the model expects |
| Sibilants sound harsh or "lispy" after downsampling | Downsampling by slicing, so high frequencies alias | Use `resample_poly` or another filtering resampler |
| Pure noise where speech should be | μ-law bytes read as plain PCM, or the other way round | Read the format code; decode μ-law |
| Distorted, crackling audio, and more errors on loud callers | Clipping from gain applied after conversion | Leave headroom; never multiply audio without checking the peak |
| Great results in testing, worse on real calls | Synthesised test audio is unnaturally crisp | Keep real, phone-quality recordings in the test set |
| The agent talks to the hold music, or never notices the caller | Energy-threshold VAD | Use a trained VAD such as Silero |

```{tip}
**Takeaways**
- Audio is just samples plus a header. Read the header; the sample rate and format in it decide everything downstream.
- A phone line keeps 300 to 3,400 Hz at 8,000 samples a second in μ-law. What it throws away is the consonants, which is where names and numbers get hard.
- Check the sample rate before anything else. Getting it wrong multiplied errors eight to fifteen times, silently.
- Resample with a filter, every time. Slicing invents sounds that were never said.
- Speech models read log-mel spectrograms: 100 frames a second, 80 or 128 bands. On phone audio a quarter of those bands are empty.
- Decide "is anyone speaking?" with a trained VAD, not a loudness threshold.
```

## Try it yourself

The companion notebook, `chapters/02-audio-for-ml`, has every cell above, the audio players, and the capture script.

1. **Hear the aliasing.** Downsample the clean call to 8 kHz by slicing and with `resample_poly`, and play both. Listen to the *s* in "is" and the *ch* in "H".
2. **HD Voice.** Modern mobile networks often carry "wideband" audio at 16 kHz with a band of about 50 to 7,000 Hz. Change `to_phone` to produce it and count how many log-mel bands come back.
3. **Tune the threshold.** Add a "hangover" to `energy_vad` that keeps it on for 300 ms after speech ends. Does it help speech recall, and what does it do to the truck?
4. **Your own voice.** Record yourself saying a ticket number on your phone, run it through `read_wav` (convert it to WAV first) and Silero, and look at the log-mel picture.

## Next

You now know exactly what arrives when Halfwheel's phone rings: 8,000 μ-law samples a second, which you resample to 16 kHz, turn into 100 log-mel frames a second, and gate with a VAD. Chapter 3 follows those frames into a speech recogniser and shows the three ways modern decoders turn them into "HW-1042".
