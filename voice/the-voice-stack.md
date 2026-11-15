---
date: "2026-11-09"
description: The whole voice-agent stack on a few pages, from the phone line to the evaluation plan, with the model chosen and the number measured at every layer, and a map of the book for different readers.
---

# The voice stack

Halfwheel Cycles is a small bike repair shop, and its phone rings while everyone's hands are covered in chain grease. Most calls are simple: is my bike ready, can I book a service, how much is a wheel true, can Priya call me back. Some are robocalls. Every missed one is a customer who might ring the shop down the road instead.

This book builds the voice receptionist that answers that phone, one layer at a time, and measures every layer on calls that sound like Halfwheel's. This first chapter is the map: what the layers are, what I chose at each one, what it measured, and where in the book to find it. Read it first for the shape of the whole, or come back to it when you want to know what something costs before you build it.

```{admonition} In this chapter
:class: note

- The whole receptionist on one page: every layer, the model I chose for it, and the number I measured.
- Three ways to build a voice agent, and why Halfwheel's is a chain of models.
- One call, timed from the caller's last word to the receptionist's first.
- What it costs, what runs where, and which chapters to read for what you're building.
```

```{admonition} New to voice agents? Words this book uses
:class: dropdown

- **Speech recognition** (ASR, or speech-to-text): a model that turns audio into text. {doc}`asr-decoders` shows how it decides what it heard.
- **Word error rate** (WER): the share of words a recogniser got wrong, counting words swapped, missed or added. 5% means one word in twenty.
- **Entities**: the words a business acts on, here ticket numbers, customer names, bike models and phone numbers. Getting these right matters more than the WER.
- **Voice activity detection** (VAD): a small model that says whether a short piece of audio contains speech.
- **End of turn**: the moment the receptionist decides the caller has finished speaking. Too early cuts them off; too late leaves them waiting.
- **Barge-in**: the caller talking over the receptionist, which should make it stop and listen.
- **Text-to-speech** (TTS): a model that turns the reply's text into a voice.
- **Language model with tools**: the model that decides what to say, and that can ask ordinary code to look up a ticket or save a booking.
- **μ-law**: the way phone networks squeeze each audio sample into one byte. {doc}`audio-for-ml` builds it from scratch.
```

## The stack on one page

A phone call passes through the same layers in every voice agent, whatever the vendor or model:

| Layer | What it does | Halfwheel's choice | What I measured | Chapter |
|---|---|---|---|---|
| **Audio** | Carries the caller's voice as numbers | 8 kHz μ-law, read and resampled correctly | the wrong sample rate took WER from 5% to 46% | {doc}`audio-for-ml` |
| **Hearing** | Turns sound into text | Parakeet-TDT, an open 0.6B transducer | decoders fail differently: every one misspelled rare names; LLM decoders can also write words nobody said | {doc}`asr-decoders` |
| **Getting names right** | Fixes the words a business depends on | light boosting and snapping to the shop's records | 96% of names, bikes, tickets and phone numbers, 90x faster than real time | {doc}`choosing-asr` |
| **Turn-taking** | Decides when the caller has finished, and handles interruptions | Silero VAD, a 1.2 s end-of-turn rule, echo cancellation | 3 of 16 hesitant callers cut off; interruptions caught in 0.06 s | {doc}`listening-live` |
| **Speaking** | Turns the reply into a voice | `speakable()` text rules, then Pocket TTS on a CPU | 65 of 66 facts heard correctly; first audio in 0.19 s | {doc}`how-tts-works` |
| **Delivery** | Makes it sound right for the moment | reference clips cast by kind of reply; phoneme fixes for names | the reference clip moved the voice; written tags barely did | {doc}`directing-a-voice` |
| **Thinking and acting** | Understands the caller and does things | Qwen3-4B with four tools, bookings confirmed in code | 11 of 16 test calls handled; prompt rules mostly ignored, code rules held | {doc}`building-the-receptionist` |
| **Screening** | Keeps robocalls from wasting time, never blocks a customer | nine yes/no questions, a learned score, rules for what never to block | 1 genuine caller of 53 blocked, 100 of 155 robocalls | {doc}`screening-calls` |
| **Evaluation** | Says whether any of it works | per-layer benches, varied callers, rules and a model judge | 11 of 16 is anywhere from 44% to 86%; a vague model judge did worse than chance | {doc}`evaluating-voice-agents` |
| **The phone line** | Connects it to real callers, at scale | media streams over WebSocket; vLLM for the model | by estimate, one 20 GB GPU had room for well over a hundred simultaneous calls on the model | {doc}`real-phone-line` |

Two things stand out from that table. First, most of the failures that matter are about **names and numbers**: a misheard bike model, a mispronounced surname, a booking under "Epha Bion" instead of Aoife Byrne. Second, the rules that held were **the ones in code**, not the ones in a prompt. Both themes come back in almost every chapter.

## Three ways to build it

The stack above is a **chain** of separate models, with text handed between them: a recogniser writes down what the caller said, a language model decides what to do and writes a reply, and a voice reads it out. Each one waits for the one before it to finish, and that is where most of the caller's wait comes from.

The alternative is **one model** that takes speech in and gives speech out, with no recogniser in front and no voice behind, and that listens while it talks. {doc}`speech-to-speech` puts Moshi, the best-known open one, on Halfwheel's phone. It sounds like a person: it picks up first, it waits through a hesitant caller's pauses, and when it did reply, it was often quicker than the chain. But it can't run a shop. With no instructions or tools, it turned a repair ticket into an airline ticket, and in 25 of 96 runs it never replied at all.

Kadence, my stand-up meeting agent, hit the same wall from the other side. An audio-in model heard "ENG-142" as "ENG-112" and called the right tool on 7 of 10 tests; a text chain got all 10. When the job is actions, text in the middle is a feature: a transcript can be checked, normalised and matched against the shop's own records before anything acts on it.

Between the two sits a **split**, which is what many systems sold as speech-to-speech really are. A speech model does the listening, the turn-taking and the voice, and hands each request to a text model that thinks and calls the tools. Chapter 10 shows how to tell the three designs apart from outside, and why a quick "time to first audio" can be an "mm-hm" rather than an answer.

So Halfwheel's receptionist is a chain, built so it can grow into a split. It borrows from the speech-to-speech models where they're strong: listening for the shape of a turn, and letting the conversation shape the voice.

Managed voice-agent platforms bundle the whole chain behind one API, and they're the fastest way to a working demo. This book is about what's inside them, so that you can choose, measure and fix each layer, whether you build it yourself or buy it.

## One call, timed

Here is where a caller's wait goes after they stop talking, from {doc}`building-the-receptionist`:

| Step | Median time |
|---|---|
| Making sure they've finished (the end-of-turn rule) | 1.2 s |
| Parakeet transcribes the turn | 0.07 s |
| The language model writes its first sentence | 0.5 s |
| Tool calls, on turns that need them | about 1 s more |
| The voice starts speaking | 0.5 s |
| **First word to the caller** | **about 2.4 s**, or 2.1 s with "one moment" played during tool calls |

People leave about 200 ms between turns {cite}`stivers2009turntaking`. The biggest single cost is the receptionist making sure the caller has finished, and that's a trade against cutting them off ({doc}`listening-live`). Serving the model with a batching engine brings its part down to about a quarter of a second ({doc}`real-phone-line`).

## What it costs

Compute is the cheap part. One GPU at well under a dollar an hour can serve many simultaneous calls, because each call needs the language model only about a seventh of the time. The bills that matter are telephony, which is most of the 2 cents per call my own robocall screen, Call Gate, pays, and hardware left running when nobody calls. {doc}`real-phone-line` has the numbers.

## What runs where

Every chapter runs on a laptop CPU from saved results in seconds, and its companion notebook reruns the models, most of them on a free Google Colab T4 GPU. The models were chosen to fit that where possible:

- **Speech recognition:** Parakeet-TDT 0.6B, with Whisper, Qwen3-ASR and Kyutai STT for comparison.
- **Voices:** Kokoro-82M and Kyutai's Pocket TTS on a CPU; Kyutai TTS, Sesame's CSM and Fish Audio's models on a GPU.
- **Language models:** Qwen3-4B, with Qwen3-1.7B and Qwen3-8B for comparison.
- **Speech-to-speech:** Moshi, which needs a 24 GB GPU such as an L4.

The callers are a mix. Halfwheel's customers are synthetic voices put through a simulated phone line, so every word and every pause is known exactly. The real-speech checks use LibriSpeech recordings, and the robocalls are real ones the US Federal Trade Commission published. Where synthetic and real speech behave differently, and they do, the chapters say so.

## How to read this book

The chapters build on each other, but you don't have to read them in order:

- **New to voice AI?** Read this chapter, then {doc}`audio-for-ml` for the foundations and {doc}`building-the-receptionist` to see the whole thing work. Come back to the middle chapters when a layer gives you trouble.
- **An ML engineer choosing models?** {doc}`asr-decoders`, {doc}`choosing-asr`, {doc}`how-tts-works` and {doc}`speech-to-speech` compare the options on the same calls.
- **Shipping a voice product?** {doc}`listening-live`, {doc}`building-the-receptionist`, {doc}`screening-calls`, {doc}`evaluating-voice-agents` and {doc}`real-phone-line` cover what goes wrong in production and how to catch it.

Every chapter opens with a Halfwheel call, puts its findings up front, ends with a table of symptoms and fixes, takeaways and exercises, and hands on to the next. The {doc}`field-guide` collects every symptom-and-fix table in one place.

```{tip}
**Takeaways**
- A voice agent is a stack: audio, hearing, turn-taking, speaking, thinking, screening, evaluation and the phone line. Each layer fails in its own way, and needs its own measurement.
- The failures that matter most are names, numbers and actions. Fix them with the business's own records and with rules in code.
- A chain of models with text in the middle is still the right shape when the job is taking actions; speech-to-speech models are already strong on turn-taking and tone.
- The caller's wait is mostly the end-of-turn decision. Compute is cheap; telephony and idle hardware are not.
```

## Next

{doc}`audio-for-ml` starts at the bottom of the stack, with the numbers a phone call is made of, and the mistakes with them that quietly ruin everything above.
