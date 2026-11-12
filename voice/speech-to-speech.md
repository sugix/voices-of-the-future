---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
date: "2027-01-11"  # quoted: myst-nb stores front matter as JSON
description: Models that take speech in and give speech out, from Spirit LM to Moshi and CSM. Moshi answers Halfwheel's callers in full duplex, CSM hears an angry caller, and Kadence shows why a chain of models still wins when actions matter.
---

# From Spirit LM to Moshi and CSM

Chapter 8's receptionist leaves a caller waiting about 2.4 seconds after they stop talking. Half of that is the end-of-turn rule: the receptionist has to be sure the caller has finished before it can hand anything to the next model in the chain. People don't talk like that. Across ten languages, the typical gap between one speaker finishing and the next starting is around 200 milliseconds {cite}`stivers2009turntaking`. We start planning our reply while the other person is still talking.

A chain of separate models, recogniser to language model to voice, can't do that, because each one waits for the last to finish. This chapter looks at the models that don't: one network that listens and speaks at the same time, with no text handed between stages. I put Moshi, the best-known open one, on Halfwheel's phone, and gave CSM, Chapter 6's conversational voice, an angry caller to reply to.

Here is where I ended up:

| Finding | Number |
|---|---|
| Moshi greets and chats in full duplex | it greeted the caller first in all {glue}`m_runs` runs, and talked over the caller in {glue}`m_over` of them |
| It isn't reliably faster on the phone, because it chooses when to speak | it started a reply within 2 s of the caller finishing in {glue}`m_quick` of {glue}`m_runs` runs, and never replied in {glue}`m_none` |
| It respects pauses | it started speaking in a caller's mid-sentence pause in {glue}`m_pause` of {glue}`m_hes` runs; Chapter 5's 0.5 s silence rule cut off 14 of 16 callers |
| It can't do the job | with no instructions and no tools, it turned a repair ticket into an airline ticket and a tune-up into "the tuner" |
| Conversation context changes the voice | after an angry caller, CSM read the same replies {glue:text}`csm_db:.0f` dB louder, faster, and with a narrower pitch range than after a calm one |
| For actions, a chain of models still wins | in Kadence, my stand-up meeting agent, a text pipeline called the right tool on 10 of 10 tests where an audio-in model managed 7 |

```{note}
Every number here comes from saved outputs, so the chapter runs on a CPU in seconds. The capture script, `sim/capture_ch10.py`, ran Moshi (7B) and CSM on one rented L4 GPU in about 30 minutes. The companion notebook reruns CSM on a free Colab T4. Moshi didn't fit a 16 GB GPU with Kyutai's PyTorch package, in float16 or with its 8-bit build, so it needs a 24 GB GPU such as an L4; on a Mac, Kyutai's MLX build is the alternative.
```

## Three steps towards one model

**Spirit LM** (Meta, 2024) {cite}`nguyen2024spiritlm` showed that a text language model can learn speech. It starts from a 7B text model and keeps training it on sequences that switch between text tokens and speech tokens (HuBERT units, plus pitch and style tokens in its Expressive version), sometimes mid-sentence. The result can continue a sentence in either form, and carries some of the meaning across. But it takes turns: it hears a whole input, then produces a whole output. And it was released for non-commercial research only, so it isn't run here.

**Moshi** (Kyutai, 2024) {cite}`defossez2024moshi` removes the turns. It models two audio streams at once, the user's and its own, as Mimi codes at 12.5 frames a second, the same codec Chapter 6 measured. Every 80 ms it reads the latest frame of the caller's audio and writes the next frame of its own. Alongside its audio it writes text, an "inner monologue" that keeps its speech coherent. Because it never stops listening, it can talk over you, say "mm-hm", or stop when you interrupt, and its designers put its theoretical latency at 160 ms, about 200 ms in practice. Its 7B backbone, Helium, is a text model trained by Kyutai, and the weights are CC BY 4.0.

**CSM** (Sesame, 2025) takes a different route. It isn't a whole conversational model; it's a voice, the one Chapter 6 tested. What makes it conversational is its input: it reads the previous turns of the conversation, both text and audio, before speaking the next one. So it can slot into Chapter 8's chain and still hear how the caller sounded.

## Moshi takes Halfwheel's calls

I played Moshi two sets of callers, three times each, over Chapter 2's phone line:

- Chapter 8's sixteen opening lines ("Hi, is my bike ready to collect? The ticket number is H W one oh four two.")
- Chapter 5's sixteen hesitant callers, who pause mid-sentence ("Hi, my ticket number is... H W one oh four two.")

Each recording starts with two seconds of silence, so Moshi can answer the phone, and ends with eight more, so it has time to reply. Moshi writes a text token whenever it starts a word, which gives the timing of everything it said.

```{code-cell} ipython3
import json
import sys

import numpy as np
from IPython.display import Audio, display

sys.path.insert(0, "sim")
from wav import load  # Chapter 2's WAV reader

moshi = json.load(open("data/ch10/moshi.json"))


def utterances(pieces, gap=0.6):
    """Group Moshi's words into utterances: a new one starts after 0.6 s without a word."""
    words = [t for t, p in pieces if p.strip()]
    out, start, last = [], None, None
    for t in words:
        if last is None or t - last > gap:
            if start is not None:
                out.append((start, last))
            start = t
        last = t
    return out + ([(start, last)] if start is not None else [])


runs = []
for cid, c in moshi["calls"].items():
    spoke = c["segments"]
    start, end = spoke[0][0], spoke[-1][1]
    pauses = [(spoke[k][1], spoke[k + 1][0]) for k in range(len(spoke) - 1)]
    for s in c["samples"]:
        u = utterances(s["pieces"])
        after = [a - end for a, _ in u if a >= end]
        runs.append({"call": cid, "kind": c["kind"], "text": s["text"],
                     "greeted": any(a < start for a, _ in u),
                     "over": any(a < end and b > start + 0.3 for a, b in u),
                     "in_pause": any(pa + 0.2 < a < pb for a, _ in u for pa, pb in pauses),
                     "reply_after": after[0] if after else None})

print(f"{'':10s} runs  greeted  talked over  reply within 2 s  no reply  started in a pause")
for kind in ["opener", "hesitant"]:
    r = [x for x in runs if x["kind"] == kind]
    print(f"{kind:10s} {len(r):4d} {sum(x['greeted'] for x in r):8d} {sum(x['over'] for x in r):11d} "
          f"{sum(x['reply_after'] is not None and x['reply_after'] <= 2 for x in r):17d} "
          f"{sum(x['reply_after'] is None for x in r):9d} {sum(x['in_pause'] for x in r):18d}")
waits = [x["reply_after"] for x in runs if x["kind"] == "opener" and x["reply_after"] is not None]
print(f"\nopeners: median wait for a new reply {np.median(waits):.1f} s")
```

```{code-cell} ipython3
:tags: [remove-cell]
from myst_nb import glue

glue("m_runs", len(runs), display=False)
glue("m_over", sum(x["over"] for x in runs), display=False)
glue("m_quick", sum(x["reply_after"] is not None and x["reply_after"] <= 2 for x in runs), display=False)
glue("m_none", sum(x["reply_after"] is None for x in runs), display=False)
glue("m_pause", sum(x["in_pause"] for x in runs if x["kind"] == "hesitant"), display=False)
glue("m_hes", sum(x["kind"] == "hesitant" for x in runs), display=False)
```

Moshi behaves like a person on the phone, though not like a receptionist. It always picks up first, with a "Hey, how can I help you?" or a "Good day, how is it going?". It often talks over the caller, which full duplex allows and turn-taking pipelines never do. And it almost never jumped into a hesitant caller's pause: on Chapter 5's callers it started speaking mid-pause in {glue}`m_pause` of {glue}`m_hes` runs, where Chapter 5's half-second silence rule cut off 14 of 16 callers. It listens to the shape of what's being said, not just to silence.

What it doesn't do is answer quickly. The 200 ms is how fast the network *can* respond; when it *does* respond is the model's choice. On phone audio it often let several seconds pass, sometimes said nothing at all until the silence went on long enough, and then filled it ("Okay, thanks for letting me know"). Part of that is the phone line: Moshi learned from wideband conversations, not 8 kHz calls with line noise, and these callers are synthetic. Still, the lesson holds. A full-duplex model removes the *pipeline's* waiting, but it doesn't make the decision to speak any easier. It moves it inside the model, where you can't tune it.

## What it says

Here are its replies to the first few openers, one run each:

```{code-cell} ipython3
by_call = {}
for x in runs:
    by_call.setdefault(x["call"], []).append(x)
for call, sample in [("c01_0", 0), ("c03_0", 2), ("c04_0", 0), ("c02_0", 1)]:
    line = by_call[call][sample]["text"].replace("\n", " ")
    print(f"{call}, run {sample + 1}: {line[:190]}\n")
for name in ["moshi_c01_0", "moshi_c04_0"]:
    x, sr = load(f"data/ch10/listen/{name}.wav")
    display(Audio(x, rate=sr))
```

A caller asks about ticket HW-1042, and Moshi prints them a train ticket. Another asks about ticket HW-1994, and Moshi explains that it's an airline ticket. Someone wants a tune-up for their Trek, and Moshi asks what they'd like to know about "the tuner". A caller asking whether their Cannondale is done is told to take a break and get some fresh air. None of this is a fault in the model. Moshi is a general conversational model with no instructions, no knowledge of the shop and no tools. It does what it was trained to do: keep a friendly conversation going. The released Moshi has no way to be told it's a bike shop's receptionist, and nothing to call to look up a repair.

That's the gap between this family of models and Chapter 8. A receptionist's value is in its actions: the booking saved, the message left, the status read from the workshop's records. Chapter 8 showed how hard it is to get even a text model to take those actions reliably. A speech-to-speech model has to do all of that while also speaking in real time, and the open ones aren't there yet.

## Conversation as context: CSM

CSM sits at the other end of the trade. It can't take a call, but it can make Chapter 8's chain sound like it heard the caller. I gave it a short conversation: an earlier receptionist turn in the voice Chapter 6 cloned, then a caller turn, then one of two replies to speak. The caller turn was the same Expresso speaker, once calm and once angry. Those recordings are about other things entirely, not bikes. What matters is how they sound. Each reply was the same words after both callers, read three times each.

```{code-cell} ipython3
csm = json.load(open("data/ch10/csm.json"))
measures = [("f0_median", "median pitch (Hz)"), ("f0_range_semitones", "pitch range (semitones)"),
            ("words_per_s", "words per second"), ("loudness_db", "loudness (dB)"), ("seconds", "length (s)")]
means = {mood: {k: np.mean([c[k] for c in csm["clips"].values() if c["mood"] == mood]) for k, _ in measures}
         for mood in ["calm", "angry"]}
print(f"{'':26s} {'after a calm caller':>20s} {'after an angry one':>20s}")
for k, label in measures:
    print(f"{label:26s} {means['calm'][k]:20.2f} {means['angry'][k]:20.2f}")
for mood in ["calm", "angry"]:
    print(f"\nafter the {mood} caller: \"{csm['replies'][0]}\"")
    x, sr = load(f"data/ch10/listen/csm_{mood}_r0_0.wav")
    display(Audio(x, rate=sr))
```

```{code-cell} ipython3
:tags: [remove-cell]
glue("csm_db", float(means["angry"]["loudness_db"] - means["calm"]["loudness_db"]), display=False)
```

The same words came out differently. After the angry caller, CSM's receptionist spoke louder, faster and with a flatter pitch, shorter and more clipped. After the calm caller it was quieter, slower and more melodic. Whether clipped is the *right* response to an angry customer is a separate question; most people would want calm and slow. But it's a lever Chapter 7 didn't have: delivery that follows the conversation, without a hand-written rule per mood. Combine it with Chapter 7's casting, choosing the reference clip by the kind of reply, and the voice can respond to both what it's saying and who it's saying it to.

## Case study: Kadence

Kadence was a voice agent I built to sit in remote stand-up meetings and update the team's Linear board while people talked: "move ENG-142 to done", "who's blocking the API ticket?". I stopped before it became a product, but it ran in real meetings, and it's the clearest comparison I have between a chain of models and an audio-native one.

Its final design was a chain, all self-hosted on one GPU: Silero for voice activity, faster-whisper large-v3-turbo, Qwen2.5-7B-Instruct on vLLM, and Kokoro, the same shape as Halfwheel's receptionist. A turn took 1.5 to 2.2 seconds warm, about the same as Chapter 8.

Early on I tried the other route: Qwen2.5-Omni-7B, which takes audio in directly, so there's no recogniser in the chain. On a small harness of 10 tool-calling tests, the audio-in model topped out at 7 of 10, even with fuzzy matching of ticket IDs. Its errors were the kind that matter: "ENG-142" heard as "ENG-112", "alex-001" as "alex-000". The text chain got all 10 right straight away, at the cost of one more step and about 150 ms. Ten tests is a small number, and an earlier run of the same Omni model on 50 utterances did much better, so this is a pointer rather than a verdict. But it's the pointer I followed: for actions, text in the middle is a feature. A transcript can be checked, normalised and snapped to known IDs, as in Chapter 4, before anything acts on it.

Two more Kadence lessons carry straight over to Halfwheel:

- **Never let the model say it acted.** Every Kadence turn had to be a tool call, including "speak" and "stay silent", and spoken confirmations came from the tool's return value, not the model's words. That's the fix for Chapter 8's receptionist that said "booked" with nothing saved.
- **Real rooms are not test benches.** Kadence's evaluation of 155 generated scenarios reached 91.6% overall after prompt hardening, with tool calls at 82% against my 90% target. In real meetings, background chatter in a café triggered unintended comments on a real ticket, and noise turned "COG-13" into "COG 31". Chapter 11 is about measuring that gap.

## Where this leaves Halfwheel

Halfwheel's receptionist stays a chain of models, because its job is actions, and actions need text in the middle. But the chain can borrow from this chapter:

1. **Turn-taking from the audio, not just silence.** Moshi's respect for mid-sentence pauses is the behaviour Chapter 5 couldn't get from a silence timer. A small turn-taking model that listens the way Moshi does, while the chain does the work, is where open systems are heading.
2. **Context for the voice.** Feed the speaking step the caller's last turn, as CSM does, so the reply sounds like it heard them.
3. **Watch the speech-to-speech models.** When open full-duplex models can follow instructions and call tools reliably, the 2.4 seconds of Chapter 8 can shrink towards a person's 200 ms. Measure them on your own bench when they arrive, with the methods of the next chapter.

## When it breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| A speech-to-speech model answers questions nobody asked | No instructions, no knowledge of the business | Use it for conversation, not for tasks; keep a chain for actions |
| It talks over callers | Full duplex: overlap is allowed | Decide whether that suits your callers; a receptionist usually shouldn't |
| It goes quiet for seconds, then fills the silence | The model chooses when to speak; phone audio isn't what it learned from | Measure response timing on your own audio before trusting a paper's latency |
| IDs come out wrong from an audio-in model | No transcript to normalise or snap | Keep text in the middle where exact values matter |
| The voice sounds the same whatever the caller's mood | The speaking step never hears the caller | Give it the conversation as context (CSM) or cast references by mood (Chapter 7) |

```{tip}
**Takeaways**
- Speech-to-speech models remove the pipeline's waiting, not the decision about when to speak. On phone audio, Moshi often waited several seconds, or never replied at all (25 of 96 runs).
- Full duplex brings human habits: picking up first, overlapping, respecting pauses. Some suit a receptionist; talking over callers usually doesn't.
- A general conversational model without instructions or tools can't run a business's phone. Its fluency makes its mistakes sound convincing.
- Conversation context changes delivery: the same words, after an angry caller, came out louder and faster.
- When the job is actions, keep text in the middle, where it can be checked. That's what Kadence found, too.
```

## Try it yourself

The companion notebook, `chapters/10-speech-to-speech`, has the saved outputs, every cell above and the capture script.

1. **Wideband callers.** Rerun Moshi on Chapter 8's callers before the phone line (regenerate them with `sim/capture_ch8.py --part voice` without `to_phone`). Does it reply sooner?
2. **Count the backchannels.** Find Moshi's "mm-hm", "yeah" and "right" while the caller is still talking. How often does it backchannel, and when?
3. **A calmer reply.** Give CSM a calm receptionist turn *after* the angry caller, then the reply. Does the earlier turn pull the delivery back?

## Next

Every chapter so far has built a bench and measured one piece. Chapter 11 turns that into a method: how to evaluate a whole voice agent, from the words it hears to the actions it takes and how it sounds, on calls that look like the ones it will really get.
