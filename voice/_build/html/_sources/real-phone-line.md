---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
date: "2027-01-25"  # quoted: myst-nb stores front matter as JSON
description: Connecting the receptionist to a real phone line. How audio streams both ways over a WebSocket, how barge-in works on the wire, how many calls one GPU can carry, and why telephony, not compute, sets the cost.
---

# Putting it on a real phone line

Everything so far has run on recordings. A real call is different in one way that matters more than any other: it doesn't wait. Audio arrives every 20 milliseconds whether the receptionist is ready or not, the caller can speak at any moment, and on a busy morning a dozen callers can ring at once. This chapter connects the receptionist to the phone network, and measures how many of those callers one GPU can serve before anyone notices.

Here is where I ended up:

| Finding | Number |
|---|---|
| Phone audio is a stream of tiny packets | 8 kHz μ-law, base64 in JSON, typically 160 bytes every 20 ms each way |
| Barge-in is one message | sending `clear` drops every buffered packet of the receptionist's reply |
| A serving engine changes the economics | Qwen3-4B on vLLM wrote its first sentence in {glue:text}`fs1:.2f` s alone and {glue:text}`fs32:.2f` s with 32 requests at once, on one 20 GB GPU |
| A call needs the model only a fraction of the time | about {glue:text}`duty:.0f`% of a call's turns are spent waiting on the language model |
| Speech recognition is nearly free | {glue:text}`asr_ms:.0f` ms of GPU per caller turn, in small batches |
| Speaking is the first bottleneck | Pocket TTS held its 0.18 s start for 4 calls at once on this pod's CPU share, and halved its speed at 8 |
| Telephony, not compute, sets the price | Call Gate pays about 2 cents a call, nearly all of it telephony; the GPU share per call-minute is a fraction of a cent |

```{note}
The load tests come from `sim/capture_ch12.py`. They replay the real model requests from Chapter 8 against Qwen3-4B on vLLM, and time Parakeet, Kokoro and Pocket TTS under load, on one rented RTX 4000 Ada GPU (20 GB). I haven't connected the receptionist to a live phone number in this chapter, because that needs your own telephony account; the protocol code below is the part that changes.
```

## How a phone call reaches your code

There are three common ways in:

- **A programmable phone number with media streams.** Twilio and similar providers answer the call and open a WebSocket to your server, then send the caller's audio and play back whatever you send. This is the quickest start, and it's how Call Gate works.
- **A SIP trunk into your own media server.** The phone carrier hands the call to a SIP endpoint, such as LiveKit's SIP service, Asterisk or FreeSWITCH, that you run. More work, more control, and usually cheaper at volume.
- **A meeting bot.** For calls that happen in Zoom or Google Meet, a bot service joins the meeting and streams each participant's audio. Kadence (Chapter 10) worked this way, which is how it got one audio track per person.

The first is the easiest to show. Twilio's bidirectional media streams send JSON messages over the WebSocket: `connected`, then `start` with the call's details, then a steady flow of `media` messages, each carrying a few milliseconds of the caller's audio, then `stop` when the caller hangs up. The audio is exactly what Chapter 2 took apart: 8 kHz, μ-law, mono, one byte per sample, base64-encoded with no WAV header.

```{code-cell} ipython3
import base64
import json
import sys

import numpy as np
from scipy import signal

sys.path.insert(0, "sim")
from wav import load, mulaw_decode  # Chapter 2's reader and decoder


def mulaw_encode(x):
    """G.711 mu-law, the inverse of Chapter 2's mulaw_decode: float samples in, one byte per sample out."""
    s = np.clip(np.round(x * 32768), -32635, 32635).astype(np.int32)
    sign = np.where(s < 0, 0x80, 0)
    m = np.abs(s) + 0x84
    exponent = np.floor(np.log2(m)).astype(np.int32) - 7
    mantissa = (m >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8)


reply, sr = load("data/ch6/listen/kokoro_r05_spoken.wav")  # a receptionist reply from Chapter 6
phone = signal.resample_poly(reply, 8000, sr)
wire = mulaw_encode(phone)
back = mulaw_decode(wire.tobytes())
snr = 10 * np.log10(np.sum(phone ** 2) / np.sum((phone - back) ** 2))
frames = [wire[i:i + 160] for i in range(0, len(wire), 160)]  # 160 bytes = 20 ms at 8 kHz
print(f"{len(phone) / 8000:.1f} s of reply = {len(wire):,} bytes = {len(frames)} packets of 20 ms; "
      f"round trip through mu-law: {snr:.1f} dB SNR")
message = {"event": "media", "streamSid": "MZ...", "media": {"payload": base64.b64encode(frames[0]).decode()}}
print(json.dumps(message)[:150], "...")
```

Every 20 ms of the receptionist's reply becomes one small JSON message, and the caller's audio arrives the same way in the other direction. The μ-law round trip keeps about 37 dB of signal over its own error, the same trade Chapter 2 measured: half the bytes of 16-bit audio, for a little noise the phone line adds anyway.

## The receptionist as a stream

On a live line, the loop from Chapter 8 becomes a set of tasks that run at the same time for each call:

1. **Receive.** Decode each `media` packet and feed it to Chapter 5's echo canceller and Silero VAD, 32 ms at a time.
2. **Detect the end of the turn,** with Chapter 5's 1.2 s rule, and hand the turn's audio to Parakeet.
3. **Think and speak.** Stream the language model's reply; as each sentence finishes, run `speakable()` (Chapter 6) and the voice, and send its audio as `media` packets. Send a `mark` after each sentence, so you know when it has actually played.
4. **Listen while speaking.** If the caller starts talking while the receptionist is, Chapter 5's barge-in rule decides. Pause on any voice; stop on a long one. Stopping is one message: `clear`, which drops every packet Twilio has buffered but not yet played. The `mark` messages then tell you which sentences the caller actually heard, which is what the conversation history should record.

Here is that skeleton as a WebSocket handler. It's the part of the receptionist that's specific to Twilio; everything it calls was built in earlier chapters.

```python
# A sketch of the Twilio side: one coroutine per call. Run it behind a public wss:// URL and point a
# Twilio number's <Connect><Stream> at it. `Turn`, `respond` and `speak` wrap Chapters 5, 8 and 6.
import base64, json
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    sid, turn, playing = None, Turn(), []          # Turn: Chapter 5's VAD, end of turn and barge-in
    async for raw in ws.iter_text():
        msg = json.loads(raw)
        if msg["event"] == "start":
            sid = msg["start"]["streamSid"]
        elif msg["event"] == "media":
            audio = mulaw_decode(base64.b64decode(msg["media"]["payload"]))
            event = turn.feed(audio, receptionist_speaking=bool(playing))
            if event == "barge_in":
                await ws.send_text(json.dumps({"event": "clear", "streamSid": sid}))
                playing.clear()                     # the caller didn't hear the rest
            elif event == "end_of_turn":
                async for sentence in respond(turn.text()):        # Chapter 8, streamed by sentence
                    for chunk in speak(sentence):                   # Chapter 6: speakable() + TTS, 8 kHz mu-law
                        payload = base64.b64encode(chunk).decode()
                        await ws.send_text(json.dumps({"event": "media", "streamSid": sid,
                                                       "media": {"payload": payload}}))
                    playing.append(sentence)
                    await ws.send_text(json.dumps({"event": "mark", "streamSid": sid,
                                                   "mark": {"name": str(len(playing))}}))
        elif msg["event"] == "mark":
            heard_up_to = int(msg["mark"]["name"])    # the caller has now heard this many sentences
        elif msg["event"] == "stop":
            break
```

Two details here are easy to get wrong. The audio you send must be raw μ-law with no WAV header, or the caller hears a click and garbage. And the replies must go out as fast as they're made, not at playback speed: Twilio buffers them and plays them in order, which is what makes `clear` work.

## How many calls can one GPU carry?

A receptionist that handles one call is a demo. The questions a business asks are how many calls at once, and what each costs. Chapter 8 ran the language model with Hugging Face transformers, one request at a time. Production serving engines like vLLM batch many requests together, so a GPU that's busy generating for one caller can generate for several more at almost the same speed.

To measure it, I replayed the 69 real requests Chapter 8's receptionist made during its sixteen calls, with their real conversation histories and tool results, against Qwen3-4B served by vLLM on one 20 GB GPU. I sent them with 1, 2, 4 and up to 64 requests in flight at the same moment:

```{code-cell} ipython3
import matplotlib.pyplot as plt

load_llm = json.load(open("data/ch12/llm_load.json"))
rows = []
for lv in load_llm["levels"]:
    r = lv["requests"]
    fs = np.array([x["first_sentence"] for x in r if x["first_sentence"]])
    ft = np.array([x["first_token"] for x in r if x["first_token"]])
    rows.append((lv["in_flight"], np.median(ft), np.median(fs), np.percentile(fs, 90), len(r) / lv["wall"]))
print("in flight  first token  first sentence (median, 90th pct)  requests/s")
for n, ft, fs, fs90, rps in rows:
    print(f"{n:9d} {ft:11.3f} s {fs:12.2f} s {fs90:7.2f} s {rps:16.1f}")

fig, ax = plt.subplots(figsize=(6.5, 3))
ax.plot([r[0] for r in rows], [r[2] for r in rows], "o-", label="median")
ax.plot([r[0] for r in rows], [r[3] for r in rows], "o--", label="90th percentile")
ax.set(xscale="log", xticks=[r[0] for r in rows], xticklabels=[str(r[0]) for r in rows],
       xlabel="requests in flight at once", ylabel="first sentence (s)",
       title="Qwen3-4B on vLLM, one 20 GB GPU")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
```

Alone, vLLM wrote the first sentence of a reply in about a quarter of a second, against roughly half a second for Chapter 8's one-at-a-time loop on an L4. With 32 requests in flight it was still under half a second, and it was answering more than twenty requests a second. Batching is the reason: the GPU's time per step barely grows as more requests share it, until memory or compute runs out.

But a call doesn't keep a request in flight all the time. Most of a call is the caller talking, the end-of-turn wait, and the receptionist's own speech. From Chapter 8's calls:

```{code-cell} ipython3
talk = json.load(open("data/ch8/talk_4b-snap-guarded.json"))["calls"]
hear = json.load(open("data/ch8/hear.json"))
model_s, turn_s = 0.0, 0.0
for turns in talk.values():
    for t in turns:
        if not t["steps"]:
            continue
        caller = hear[t["clip"]]["speech_end"] - 0.3
        thinking = sum(s["timing"]["total"] for s in t["steps"])
        speaking = len(t["reply"].split()) / 2.6  # Chapter 6: about 2.6 words a second from the voice
        model_s += thinking
        turn_s += caller + (hear[t["clip"]]["endpoint_wait"] or 1.2) + thinking + speaking
duty = model_s / turn_s
print(f"the language model is busy for {duty:.0%} of a call")
for n, ft, fs, fs90, rps in rows:
    if n in (16, 32, 64):
        print(f"{n} requests in flight at once is roughly {n / duty:.0f} simultaneous calls")
```

```{code-cell} ipython3
:tags: [remove-cell]
from myst_nb import glue

glue("fs1", float(rows[0][2]), display=False)
glue("fs32", float([r for r in rows if r[0] == 32][0][2]), display=False)
glue("duty", float(100 * duty), display=False)
```

The language model works for only a small fraction of each call. Even after allowing for bursts, with several callers finishing their turns at the same moment, one 20 GB GPU running vLLM has room for well over a hundred simultaneous calls on the language model alone. That's an estimate from replayed requests, not a test with real calls, so treat it as an upper bound to test against, not a promise.

The other two models on the GPU need far less:

```{code-cell} ipython3
asr = json.load(open("data/ch12/asr_load.json"))
print("Parakeet-TDT, Chapter 8's 65 caller turns:")
for a in asr:
    print(f"   batch {a['batch']:3d}: {1000 * a['wall'] / a['clips']:4.0f} ms of GPU per turn, "
          f"{a['audio_s'] / a['wall']:5.0f}x real time")
tts = json.load(open("data/ch12/tts_load.json"))
print("\nKokoro on the GPU, first audio of a reply's first sentence:")
for k in tts["kokoro_gpu"]:
    first = [x for x, _ in k["requests"]]
    print(f"   {k['parallel']} at once: median {np.median(first):.2f} s, 90th percentile {np.percentile(first, 90):.2f} s")
print("\nPocket TTS on CPU processes (2 threads each), first audio:")
for k in tts["pocket_cpu"]:
    print(f"   {k['processes']} at once: median {np.median([x[0] for x in k['requests']]):.2f} s, "
          f"{np.median([x[2] / x[1] for x in k['requests']]):.1f}x real time each")
```

```{code-cell} ipython3
:tags: [remove-cell]
glue("asr_ms", float(min(1000 * a["wall"] / a["clips"] for a in asr)), display=False)
```

Parakeet costs a few tens of milliseconds of GPU per caller turn, so speech recognition is effectively free. Speaking is where the limits show first. Kokoro on the GPU started in 0.09 s alone, but slowed sharply with eight requests at once from my simple threads; a serving setup that batches TTS requests, as vLLM does for text, would do much better. Pocket TTS on the CPU held its 0.18 s start with four calls at once, then ran out of this pod's CPU share at eight, dropping to real time. So the first thing to scale on a real deployment is the voice, not the brain: more CPU cores for Pocket TTS, or batched TTS on the GPU.

## What it costs

At these numbers, compute is the cheap part. The GPU for this test cost $0.28 an hour. Spread over even twenty simultaneous calls, that's a fraction of a cent per call-minute, before counting the CPU for the voice. The real costs are elsewhere:

- **Telephony.** Call Gate, which uses a programmable phone number much like the one above, pays about 2 cents per screened call, and nearly all of that is telephony, not the models.
- **Idle hardware.** Kadence cost about $2.50 per meeting-hour while running: a GPU at about $1.86 an hour and meeting transport at about $0.50. Its most expensive mistake was a GPU left running for 59 days with no meetings, about $2,650. A receptionist that answers the phone at 3 a.m. needs a GPU at 3 a.m., but it doesn't need one all night at full size.

The practical rule: size the GPU for your busiest hour, scale down or share it when the phones are quiet, and look at the telephony bill first when you want to save money.

## Before you switch it on

A real line brings obligations a recording doesn't:

- **Tell callers they're talking to an automated assistant, and that the call may be recorded,** if you record. Call Gate's first words are "This call may be recorded." The rules on this vary by country and state, so check the ones that apply to you.
- **Always offer a way out to a person:** a message, a callback or a transfer. Chapter 8's receptionist takes messages when it can't help; on a live line that's the safety net for every failure in this book.
- **Have a fallback for every component.** If the language model is slow, play Chapter 8's filler phrase; if it's down, take a message; if speech recognition fails, ask the caller to repeat. A caller should never hear silence.
- **Log every call,** as Chapter 11 recommends, and run new versions in shadow mode first, as Call Gate did.

## When it breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| The caller hears clicks or noise from the receptionist | A WAV header or the wrong format in outbound `media` | Send raw 8 kHz μ-law bytes, base64-encoded, no header |
| The receptionist keeps talking after the caller interrupts | Audio already buffered at the provider | Send `clear` on barge-in; use `mark` to know what was heard |
| The history says the receptionist said things the caller never heard | Replies recorded when generated, not when played | Update the history from `mark` messages |
| Replies get slow when several calls arrive together | One-at-a-time generation | A batching server (vLLM or similar) for the language model |
| The voice gets slow before the model does | TTS without batching, or too few CPU cores | Batch TTS on the GPU, or more cores for CPU TTS |
| The bill is high on a quiet month | Idle GPUs | Scale to traffic; the GPU is cheap only while it's busy |

```{tip}
**Takeaways**
- On the wire, a call is 20 ms packets of 8 kHz μ-law each way, and barge-in is a single `clear` message.
- Stream everything: send each sentence as soon as it's spoken, and use `mark` to track what the caller actually heard.
- A batching server lets one GPU handle many calls, because each call needs the language model only a small fraction of the time.
- Speech recognition is nearly free; the voice is the first thing to run out of capacity.
- Telephony and idle hardware cost more than inference. Size for the busy hour and scale down when it's quiet.
- On a live line: disclose, record responsibly, offer a person, and never leave the caller in silence.
```

## Try it yourself

The companion notebook, `chapters/12-real-phone-line`, has the load-test results, every cell above, the capture script and the WebSocket sketch.

1. **Your own number.** With a Twilio trial account, point a number's `<Connect><Stream>` at the sketch above, behind a tunnel, and call it. Start with an echo: send each `media` payload straight back.
2. **Measure barge-in.** Time from the caller starting to speak to your `clear` message, and from `clear` to silence on your phone.
3. **Load test your own model.** Rerun `sim/capture_ch12.py --part llm` with Qwen3-8B. How many requests in flight before the first sentence passes one second?

## Next

This was the last step of the build: Halfwheel's receptionist can hear, think, speak, screen, be evaluated, and answer a real phone. Chapter 1, which opens the book, puts the whole stack on one page, with every number from these chapters, so that a reader starting there knows what each piece costs before building it.
