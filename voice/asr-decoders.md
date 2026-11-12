---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
date: "2026-11-16"  # quoted: myst-nb stores front matter as JSON
description: CTC, transducers and LLM decoders, built from scratch in numpy, and what each one means for speed, streaming and getting ticket numbers right.
---

# How speech recognition decoders read audio

A caller rings Halfwheel Cycles and says: *"Hi, is my bike ready? The ticket number is HW-1042."* Before the receptionist can look anything up, a speech recognition model has to turn that sound into text. The part of the model that makes the final decision about which characters or words it heard is the **decoder**, and I've found that choosing the decoder matters more than most model comparisons let on.

Here is what this chapter shows, so you can skip to the part you need:

| | CTC | Transducer (RNN-T, TDT) | LLM decoder |
|---|---|---|---|
| Example models | Parakeet-CTC | Parakeet-RNNT, Parakeet-TDT | Qwen3-ASR, Whisper |
| Uses the text it already wrote | No | Yes | Yes |
| Sequential steps after the encoder | One parallel pass | One per frame plus one per token (fewer with TDT) | One per token, each through a large model |
| Typical failure | Misspells names and codes it has never seen | Fewer of those, but still literal | Writes fluent text that was never said, especially over silence |
| Streams? | Yes, with a streaming encoder | Yes, with a streaming encoder | Possible, but harder |

All three share the same first half: an **encoder** that turns audio into a sequence of feature vectors, one every 80 ms for the FastConformer encoders in NVIDIA's Parakeet models {cite}`rekesh2023fastconformer`. They differ in how they turn those vectors into text. Every algorithm below runs live in numpy on the page, and at the end the same Halfwheel call goes through three real models.

```{note}
Code for this chapter runs on any laptop. The real-model section at the end needs a GPU; the companion notebook runs it on a free Colab T4: [github.com/sugix/voices-of-the-future](https://github.com/sugix/voices-of-the-future).
```

## The problem: long audio, short text, no alignment

A 10 second call gives the encoder 125 frames. The transcript is about 25 words, or roughly 30 tokens for a subword tokenizer. Nobody labels which frames belong to which character, and nobody wants to. Training data is just audio and its transcript.

So the decoder has to learn a mapping between a long sequence and a short one without being told where anything lines up. Each of the three decoders solves that differently.

```{code-cell} ipython3
import itertools

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(1042)

# A character vocabulary. Index 0 is the CTC "blank".
VOCAB = ["_"] + list("abcdefghijklmnopqrstuvwxyz0123456789 -")
IDX = {c: i for i, c in enumerate(VOCAB)}
BLANK = 0
FRAME_S = 0.08  # 80 ms per encoder frame
```

## CTC: one guess per frame, then clean up

Connectionist Temporal Classification {cite}`graves2006ctc` makes one prediction per encoder frame. Each prediction is either a character or a special **blank** that means "nothing new here". To get text, you apply one rule: **merge repeated symbols, then remove blanks.**

```{code-cell} ipython3
def ctc_collapse(path):
    """Merge consecutive repeats, then drop blanks."""
    out, prev = [], None
    for s in path:
        if s != prev and s != "_":
            out.append(s)
        prev = s
    return "".join(out)


for p in ["hh_i", "hhhiii", "bell", "bel_ll"]:
    print(f"{p!r:>10} -> {ctc_collapse(p)!r}")
```

The blank does two jobs. It lets the model say nothing while a sound is still going on, and it separates genuine double letters: without a blank between them, the two l's in "bell" merge into one.

Here is a fake set of frame probabilities for our caller's ticket number, shaped like what a real CTC head produces: a sharp peak per frame, mostly on blank.

```{code-cell} ipython3
def fake_posteriors(path, confidence=6.0, noise=0.8):
    """Frame-by-frame probabilities that peak on the given path."""
    logits = rng.normal(0, noise, size=(len(path), len(VOCAB)))
    for t, s in enumerate(path):
        logits[t, IDX[s]] += confidence
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def greedy_ctc(probs):
    return ctc_collapse([VOCAB[i] for i in probs.argmax(axis=1)])


path = "__hh__ww___ ___11__00___44__22___"
probs = fake_posteriors(path)
print(len(probs), "frames ->", repr(greedy_ctc(probs)))

shown = ["_", "h", "w", " ", "1", "0", "4", "2"]
fig, ax = plt.subplots(figsize=(9, 2.6))
ax.imshow(probs[:, [IDX[c] for c in shown]].T, aspect="auto", cmap="magma")
ax.set_yticks(range(len(shown)), ["blank" if c == "_" else repr(c) for c in shown])
ax.set_xlabel("encoder frame (80 ms each)")
ax.set_title("CTC frame probabilities for 'hw 1042'")
plt.tight_layout()
```

Greedy decoding takes the most likely symbol at every frame. Because each frame is scored independently, a GPU computes all of them at once. That is why CTC is the fastest of the three decoders.

The catch is in that word *independently*. The CTC head never looks at what it already wrote. It cannot know that "hw" is usually followed by four digits at Halfwheel. That knowledge has to come from somewhere else, which is the next section.

### How CTC learns without alignments

Many frame paths collapse to the same text: `hh_i`, `h_ii` and `_hi_` all mean "hi". CTC training maximises the **total probability of every path that collapses to the right transcript**. Adding up all of them looks impossible, but the forward algorithm does it in time proportional to frames times characters {cite}`hannun2017distill`.

Here it is, checked against brute force on a problem small enough to enumerate:

```{code-cell} ipython3
def ctc_log_prob(probs, label_ids):
    """log P(label | audio), summed over every alignment (forward algorithm)."""
    ext = [BLANK]
    for c in label_ids:
        ext += [c, BLANK]
    S, T = len(ext), len(probs)
    logp = np.log(probs + 1e-12)
    alpha = np.full((T, S), -np.inf)
    alpha[0, 0] = logp[0, ext[0]]
    alpha[0, 1] = logp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            a = alpha[t - 1, s]
            if s >= 1:
                a = np.logaddexp(a, alpha[t - 1, s - 1])
            if s >= 2 and ext[s] != BLANK and ext[s] != ext[s - 2]:
                a = np.logaddexp(a, alpha[t - 1, s - 2])
            alpha[t, s] = a + logp[t, ext[s]]
    return np.logaddexp(alpha[-1, -1], alpha[-1, -2])


SMALL = ["_", "a", "b"]
small = rng.dirichlet(np.ones(3), size=5)  # 5 frames over {blank, a, b}
paths = [p for p in itertools.product(range(3), repeat=5)
         if ctc_collapse([SMALL[i] for i in p]) == "ab"]
brute = sum(np.prod(small[np.arange(5), p]) for p in paths)
print(f"{len(paths)} of {3**5} paths collapse to 'ab'")
print(f"brute force: {brute:.6f}   forward algorithm: {np.exp(ctc_log_prob(small, [1, 2])):.6f}")
```

The two numbers match. Real CTC loss is exactly this, in log space, batched on a GPU.

## Fixing "HW-1042" with what the shop already knows

Now make the audio harder. Suppose the caller's "four" was muffled, so at those frames the model gives "1" slightly more probability than "4":

```{code-cell} ipython3
noisy = fake_posteriors(path)
for t, s in enumerate(path):
    if s == "4":
        row = np.full(len(VOCAB), 0.004)
        row[IDX["1"]], row[IDX["4"]] = 0.50, 0.42
        noisy[t] = row / row.sum()

heard = greedy_ctc(noisy)
print("greedy:", repr(heard))
```

Greedy decoding writes `hw 1012`. Halfwheel has no such ticket. But the shop knows which tickets are open, and the forward algorithm can score any candidate text against the audio. So instead of trusting the greedy guess, I score every open ticket and keep the best:

```{code-cell} ipython3
open_tickets = ["hw 1042", "hw 1024", "hw 2041", "hw 1402"]
scores = {t: ctc_log_prob(noisy, [IDX[c] for c in t]) for t in open_tickets}
for t, s in sorted(scores.items(), key=lambda kv: -kv[1]):
    print(f"{t}: log P = {s:8.2f}")
print("best open ticket:", max(scores, key=scores.get))
```

This is the first real piece of the receptionist: **when the caller says something from a known list, check the audio against the list instead of trusting the transcript.** Production toolkits do a fuller version of the same idea with beam search, an n-gram language model or word boosting, which adds bias toward phrases you supply. In Kadence, my voice agent for Linear, café noise turned "COG-13" into "COG 31" in the transcript. A check against the open issues, like this one, would have flagged it before the agent acted.

```{warning}
Rescoring always returns *some* ticket. If the caller reads out a ticket that is not open, you still get the closest one. Compare the best score with the greedy path's score, and ask the caller to confirm when they are far apart.
```

## Transducers: let the decoder read its own output

A transducer {cite}`graves2012rnnt` adds the context CTC lacks. Besides the encoder, it has a small **prediction network** that reads the tokens emitted so far, and a **joint network** that combines the current audio frame with that text history. At each step the joint network picks either a token or a blank:

- a **token** extends the transcript, and the decoder stays on the same audio frame;
- a **blank** moves to the next audio frame.

So the decoder walks through a grid of (audio frame, tokens so far), and its decisions depend on what it has already written. That is why transducers usually beat CTC on names and rare words. The price is that the walk is sequential: greedy RNN-T decoding needs one decision per frame plus one per token.

**TDT**, the Token-and-Duration Transducer behind Parakeet-TDT {cite}`xu2023tdt`, also predicts **how many frames to skip**, from 0 to 4 in the released models. Silence and long vowels no longer cost a decision per frame. Here are the decisions each decoder needs for the same audio, using a perfect oracle so only the counting differs:

```{code-cell} ipython3
def rnnt_steps(token_frames, T):
    return len(token_frames) + T  # every token, plus one blank per frame


def tdt_steps(token_frames, T, max_dur=4):
    steps, t, i = 0, 0, 0
    while t < T:
        if i < len(token_frames) and token_frames[i] == t:
            i += 1  # emit a token at this frame
        nxt = token_frames[i] if i < len(token_frames) else T
        steps += 1
        t += min(max_dur, nxt - t)  # duration predicted with the decision
    return steps


seconds = np.arange(1, 31)
rows = []
for sec in seconds:
    T = int(sec / FRAME_S)
    U = int(sec * 3)  # about 3 subword tokens per second of speech
    frames = np.linspace(0, T - 1, U).astype(int)
    rows.append((1, rnnt_steps(frames, T), tdt_steps(frames, T), U + 1))
ctc, rnnt, tdt, llm = np.array(rows).T

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(seconds, rnnt, label="RNN-T (frames + tokens)")
ax.plot(seconds, tdt, label="TDT (skips up to 4 frames)")
ax.plot(seconds, llm, label="LLM decoder (one per token)")
ax.plot(seconds, ctc, label="CTC (one parallel pass)")
ax.set_xlabel("audio length (s)")
ax.set_ylabel("sequential decoder steps")
ax.legend(frameon=False)
plt.tight_layout()
print(f"10 s of audio: RNN-T {rnnt[9]}, TDT {tdt[9]}, LLM {llm[9]}, CTC {ctc[9]}")
```

Step counts are not wall-clock time. A transducer step is a small joint network; an LLM decoder step runs a model with a billion or more parameters. But the counts explain why Parakeet-TDT is so fast in practice: it removes most of the per-frame decisions while keeping the text history.

## LLM decoders: a language model that listens

The third design hands the encoder's output to a language model. Qwen3-ASR projects audio embeddings into the input of a Qwen3 language model, which then generates the transcript one token at a time, attending to all of the audio and everything it has written so far {cite}`qwen2025asr`. Whisper is an earlier, smaller example of the same encoder-decoder idea.

What you gain is context in both directions. The decoder can use the whole sentence to settle an unclear word, handle punctuation and casing naturally, and cover many languages: Qwen3-ASR lists 30 languages plus 22 Chinese dialects. What you pay:

- **One heavy step per token.** A 15-token sentence needs 15 passes through the language model.
- **Fluency beyond the audio.** A language model is trained to produce likely text. Over silence or noise it can produce a likely sentence that nobody said. In Kadence, Whisper once appended a transcription service's credit line to a real sentence. That came from its training data, not from the meeting.
- **Harder streaming.** The decoder wants to see the whole utterance. Qwen3-ASR does support streaming, but only through its vLLM backend.

## What streaming really costs

For live captions or a receptionist that answers quickly, the decoder is only part of the delay. A streaming encoder processes audio in chunks and may wait for some **look-ahead** (future audio) before it commits to a frame. The last word of a chunk waits for the chunk to fill, plus the look-ahead, plus compute. The settings below are illustrative, not measurements:

```{code-cell} ipython3
def worst_case_delay_ms(chunk_ms, lookahead_ms, compute_ms):
    return chunk_ms + lookahead_ms + compute_ms


for chunk, look, comp in [(80, 0, 15), (160, 80, 15), (480, 480, 20), (1000, 1000, 25)]:
    print(f"chunk {chunk:4d} ms + look-ahead {look:4d} ms + compute {comp:2d} ms "
          f"= {worst_case_delay_ms(chunk, look, comp):5d} ms")
```

More look-ahead usually means better accuracy, because the encoder hears the end of a word before deciding on it. CTC and transducers both stream well when the encoder is built for it. Chapter 5 measures these trade-offs on real streaming models.

## Bonus: CTC tells you when each word was said

The same grid the forward algorithm walks can find the **single most likely alignment** instead of summing over all of them. That gives every character, and therefore every word, a start and end time. This is **forced alignment**, and it is how karaoke-style highlighting and word-level audio editing work.

```{code-cell} ipython3
def ctc_align(probs, label_ids):
    """Most likely alignment (Viterbi). Returns (first, last) frame per label."""
    ext = [BLANK]
    for c in label_ids:
        ext += [c, BLANK]
    S, T = len(ext), len(probs)
    logp = np.log(probs + 1e-12)
    score = np.full((T, S), -np.inf)
    back = np.zeros((T, S), dtype=int)
    score[0, 0], score[0, 1] = logp[0, ext[0]], logp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            cands = [(score[t - 1, s], s)]
            if s >= 1:
                cands.append((score[t - 1, s - 1], s - 1))
            if s >= 2 and ext[s] != BLANK and ext[s] != ext[s - 2]:
                cands.append((score[t - 1, s - 2], s - 2))
            best, arg = max(cands)
            score[t, s], back[t, s] = best + logp[t, ext[s]], arg
    s = S - 1 if score[-1, S - 1] >= score[-1, S - 2] else S - 2
    states = [0] * T
    for t in range(T - 1, -1, -1):
        states[t], s = s, back[t, s]
    spans = {}
    for t, s in enumerate(states):
        if s % 2 == 1:
            k = s // 2
            a, b = spans.get(k, (t, t))
            spans[k] = (min(a, t), max(b, t))
    return [spans[k] for k in range(len(label_ids))]


text = "hw 1042"
spans = ctc_align(probs, [IDX[c] for c in text])
words, start = [], 0
for i, c in enumerate(text + " "):
    if c == " ":
        first, last = spans[start][0], spans[i - 1][1]
        words.append((text[start:i], first * FRAME_S, (last + 1) * FRAME_S))
        start = i + 1
for w, a, b in words:
    print(f"{w:>5}: {a:.2f} s to {b:.2f} s")
```

I use this in Reprise, a voice-direction studio I'm building on Fish Audio's Drama 3. Word timings from forced alignment let you re-record a single word of a line instead of the whole take. Chapter 7 comes back to it.

## On real models

Toy probabilities only go so far. I ran all six Halfwheel calls through one model of each kind, both as clean audio and as the 8 kHz phone version. The calls are generated with Kokoro, so you can reproduce them exactly. The code is below, and the companion notebook runs it on a free Colab T4.

```python
# pip install "nemo_toolkit[asr]" qwen-asr soundfile
import nemo.collections.asr as nemo_asr
import torch
from qwen_asr import Qwen3ASRModel

wav = "data/calls/phone/c01_status.wav"

for name in ["nvidia/parakeet-ctc-0.6b", "nvidia/parakeet-tdt-0.6b-v2"]:
    model = nemo_asr.models.ASRModel.from_pretrained(name).cuda().eval()
    print(name, "->", model.transcribe([wav], verbose=False)[0].text)

qwen = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.float16,  # a T4 has no bfloat16 support
    device_map="cuda:0",
)
print("Qwen3-ASR ->", qwen.transcribe(audio=wav, language=None)[0].text)
```

Here are the three calls with the most to teach, over the phone line:

| What was said | Parakeet-CTC | Parakeet-TDT | Qwen3-ASR |
|---|---|---|---|
| ...the ticket number is HW-1042. | ...the ticket number is h w one oh four two | ...The ticket number is HW1042. | ...The ticket number is H W 1042. |
| ...a tune-up for my Trek Domane SL 5... | ...my trek domain sl f... | ...my Trek Domain SL5... | ...my Trek Domain SL5... |
| ...It's Daniel Okafor, on 415-555-0192. | ...it's daniel oaffa on four one five five five five oh one nine two | ...It's Daniel Okafer on 415-555-0192. | ...It's Daniel O'Kaffer on four one five five five five o one nine two. |

The other three calls, about a price, a booking change and a spam offer, came out right from all three models, apart from formatting such as "ten" versus "10" and punctuation. Here is what I take from the hard ones:

- **All three got the ticket, but each wrote it differently.** CTC writes the spoken form, TDT writes `HW1042`, and Qwen3-ASR writes `H W 1042`. Before you compare a transcript with your list of open tickets, convert both to one written form. Chapter 4 covers that conversion, called inverse text normalization.
- **No model knew the Trek Domane.** All three wrote "Domain", the common English word. Even Qwen3-ASR's language model did not help here: a rare brand name loses to a common word. This is exactly what rescoring against the shop's own catalogue, or word boosting, is for.
- **Nobody spelled Okafor correctly, and every model spelled it differently.** For names, the fix is a conversation, not a better model: the receptionist should ask the caller to spell it, or check it against caller ID.
- **CTC was the only one to lose a word on the phone line.** It heard "SL five" as "sl f". The two decoders that read their own output kept it.

These are six synthetic calls, not a benchmark. Chapter 4 measures word error rates on real human speech, which is harder.

Speed, on an RTX 4000 Ada with one call at a time:

| Model | Decoder | Seconds per call | Faster than real time | Peak GPU memory |
|---|---|---|---|---|
| Parakeet-CTC 0.6B | CTC | 0.11 | 59x | small |
| Parakeet-TDT 0.6B v2 | TDT | 0.10 | 66x | small |
| Qwen3-ASR 1.7B (fp16) | LLM | 0.86 | 8x | 7.5 GB |

On calls this short, fixed overhead hides the difference between CTC and TDT. The LLM decoder is about eight times slower, which is what the step counts earlier predicted. All three fit in a free T4's 16 GB.

Finally, here are the real CTC probabilities from Parakeet-CTC for the ticket call, saved from that run. Its vocabulary has 1,024 subword tokens plus the blank, and the decoding rule is the same one you wrote at the start of the chapter:

```{code-cell} ipython3
d = np.load("data/ch3/parakeet_ctc_c01.npz")
lp, vocab = d["logprobs"].astype(np.float32), list(d["vocab"])
blank = len(vocab) - 1
best = lp.argmax(axis=1)

tokens, prev, emitted = [], None, []
for t, i in enumerate(best):
    if i != prev and i != blank:
        tokens.append(vocab[i])
        emitted.append((t, vocab[i]))
    prev = i
print(f"{len(lp)} frames, {np.mean(best == blank):.0%} of them blank")
print("".join(tokens).replace("▁", " ").strip())

p = np.exp(lp)
times = np.arange(len(p)) * FRAME_S
fig, ax = plt.subplots(figsize=(9, 2.8))
ax.plot(times, p[:, blank], label="blank", color="0.6")
ax.plot(times, np.delete(p, blank, axis=1).max(axis=1), label="best token")
for t, tok in emitted:
    ax.annotate(tok.replace("▁", ""), (t * FRAME_S, 1.02), ha="center", fontsize=7, rotation=60)
ax.set_ylim(0, 1.25)
ax.set_xlabel("seconds")
ax.set_ylabel("probability")
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
```

Two thirds of the frames are blank, and each token appears as a single sharp spike, just like the toy example. Real CTC heads really do work this way.

## When it breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| Codes come out close but wrong ("hw 1012") | CTC has no text context, and the code is rare | Rescore against the list you already have, or use word boosting |
| Brand names become common words ("Domane" to "Domain") | Every decoder prefers frequent words | Rescore against your catalogue, or boost those words |
| Surnames come out differently every time | Rare names, with no context to pin them down | Ask the caller to spell it, or check caller ID |
| The same ticket is written three ways | Each model formats numbers differently | Convert transcripts and your list to one written form before comparing |
| Doubled or missing letters | Blank handling in a hand-written decoder | Merge repeats *before* removing blanks |
| A sentence appears over silence or hold music | LLM decoder producing likely text | Run VAD first, and filter on no-speech probability |
| Accuracy drops on phone calls | 8 kHz audio into a 16 kHz model | Resample correctly, and test on phone-quality audio (Chapter 2) |
| Captions lag behind the speaker | Chunk size plus look-ahead | Smaller chunks, less look-ahead, or a streaming encoder |

## What the receptionist has now

The receptionist can now take a ticket number from a noisy call and check it against the shop's open tickets, instead of trusting the transcript. It has also seen real models fail on a bike model and a surname, which tells me where to look hardest when I choose a model.

```{tip}
**Takeaways**
- The decoder, not just the model size, decides how fast a model is, whether it can stream, and what context it uses.
- CTC is the fastest and the most literal. A transducer adds text history for almost nothing. An LLM decoder adds the most context and costs the most, including text that was never said.
- When the caller says something from a list you already have, check the audio against the list instead of trusting the transcript.
- Every model failed on the same things: a rare brand name and a surname. Those are the words a business cares about most.
```

## Try it yourself

Everything in this chapter runs in the companion notebook, `chapters/03-asr-decoders`, on a laptop or a free Colab T4.

1. **Break the decoder.** In `fake_posteriors`, lower `confidence` until greedy decoding gets the clean ticket wrong. How much worse does the audio have to get before rescoring fails too?
2. **Close tickets.** Add `"hw 1043"` to `open_tickets`. Rescoring still picks the right one, but by how much? That margin is what you would use to decide when to ask the caller to repeat.
3. **Your own voice.** Record yourself reading a Halfwheel ticket number and run it through Parakeet-TDT and Qwen3-ASR. Then try it over speakerphone.

## Next

All three models wrote "Trek Domain". In Chapter 4 I find out whether that was bad luck or a pattern. I measure six models on 30 Halfwheel calls and 80 real human recordings, clean and over a phone line, score them on the words the shop depends on rather than on overall word error rate, and teach them the shop's vocabulary.
