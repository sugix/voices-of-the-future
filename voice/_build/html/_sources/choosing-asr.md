---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
date: "2026-11-23"  # quoted: myst-nb stores front matter as JSON
description: Six speech recognition models, 30 bike-shop calls and 80 real recordings, scored on the words a business actually depends on.
---

# Choosing speech recognition for your callers

Imagine Grace Adeyemi rings Halfwheel Cycles about her Canyon Grizl, and the transcript reads "Grace a Daymi" and "Canyon Grizzly". Forty-eight of her fifty words are right, so the word error rate looks excellent. The two that are wrong are the only two the shop needed.

That is the whole problem with choosing a speech recognition model from a leaderboard. In this chapter I build a small test bench for Halfwheel, run six models through it, and choose one. Along the way you get three tools you can reuse for any voice product: a word error rate you understand line by line, a normalizer that stops formatting from polluting it, and an **entity score** that measures what your business actually cares about.

Here is where I ended up, so you can skip to the part you need:

| Finding | Number |
|---|---|
| Word error rates on Halfwheel's phone calls were close together | 4.2% to 6.4% across six models |
| The share of shop-critical words they got right was not | 54% to 65% |
| Telling the model the shop's words helped most | Whisper 65% to 77%, Parakeet-TDT 62% to 71% |
| Snapping transcripts to the shop's catalogue helped even more | up to 98% |
| Too much hinting backfired | a boosted model wrote a *different customer's name* |
| Halfwheel's choice for live calls | Parakeet-TDT, light boosting and catalogue snapping: 96% of entities, 90x faster than real time |

In Chapter 3, every model failed on the same bike model and the same surname. This chapter turns that anecdote into a measurement, and then fixes most of it without changing a single model weight.

```{note}
The scoring code runs anywhere. The model runs need a GPU, and the companion notebook does them on a free Colab T4: [github.com/sugix/voices-of-the-future](https://github.com/sugix/voices-of-the-future), `chapters/04-choosing-asr`.
```

## Word error rate, from scratch

Word error rate (WER) counts the fewest word **substitutions**, **deletions** and **insertions** that turn the model's transcript into the reference, divided by the number of reference words. It is an edit distance over words, and it fits in a dozen lines:

```{code-cell} ipython3
import json
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def edit_ops(ref, hyp):
    """Minimum (substitutions, deletions, insertions) turning ref into hyp."""
    R, H = len(ref), len(hyp)
    d = np.zeros((R + 1, H + 1), dtype=int)
    d[:, 0], d[0, :] = np.arange(R + 1), np.arange(H + 1)
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            d[i, j] = min(d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1, j] + 1, d[i, j - 1] + 1)
    i, j, ops = R, H, {"sub": 0, "del": 0, "ins": 0}
    while i or j:  # walk back to count each kind of error
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops["sub"] += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops["del"] += 1
            i -= 1
        else:
            ops["ins"] += 1
            j -= 1
    return ops


def wer(ref, hyp):
    ref, hyp = ref.split(), hyp.split()
    ops = edit_ops(ref, hyp)
    return sum(ops.values()) / max(len(ref), 1), ops


ref = "grace adeyemi about her canyon grizl"
hyp = "grace a daymi about her canyon grizzly"
print(wer(ref, hyp))
```

Two wrong words became three errors: "adeyemi" turned into two words, which counts as one substitution plus one insertion. WER can exceed 100%, and it treats every word as equally important. Both facts matter later.

## Normalize before you score

Here is one real transcript from Chapter 3, written three ways by three models. All three heard the ticket correctly:

```{code-cell} ipython3
cap = json.load(open("data/ch3/capture_ch3.json"))
truth = "Hi, is my bike ready? The ticket number is HW-1042."
for model, res in cap["results"].items():
    hyp = res["c01_status/phone"]["text"]
    print(f"{model:30} WER {wer(truth.lower(), hyp.lower())[0]:5.0%}   {hyp}")
```

Measured naively, the model that wrote out every digit looks terrible, and even the best one loses points for "HW1042" against "HW-1042". None of that is a recognition error. It is formatting. So before scoring, I turn every transcript, and every reference, into one plain spoken-style form: lowercase, no punctuation, digits as digit words, letters spelled apart.

```{code-cell} ipython3
DIGITS = "zero one two three four five six seven eight nine".split()
SMALL = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
SMALL["oh"] = 0
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}


STOP = set("a i is it on to my me of in at be do so up or an as by if no we he us am hi ok".split())


def normalize(text):
    """One canonical form for scoring: 'HW-1042' and 'h w one oh four two' both become 'h w 1 0 4 2'."""
    t = text.lower().replace("-", " ").replace("'", "")
    toks = re.sub(r"[^a-z0-9 ]", " ", t).split()
    out, i = [], 0
    while i < len(toks):
        tok = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if tok in TENS and nxt in SMALL and 0 < SMALL[nxt] < 10:
            out += list(str(TENS[tok] + SMALL[nxt]))  # twenty one -> 2 1
            i += 2
            continue
        if tok.isdigit() or re.fullmatch(r"[a-z]+\d+|\d+[a-z]+", tok):
            out += list(tok)  # 1042 -> 1 0 4 2, hw1042 -> h w 1 0 4 2, sl5 -> s l 5
        elif tok in SMALL:
            out += list(str(SMALL[tok]))  # one oh four two -> 1 0 4 2, ten -> 1 0
        elif tok == "o" and ((out and out[-1].isdigit()) or nxt in SMALL or nxt.isdigit()):
            out.append("0")  # 'seven o' sounds exactly like 'seven oh'
        elif tok in TENS:
            out += list(str(TENS[tok]))
        elif len(tok) <= 2 and tok not in STOP:
            out += list(tok)  # 'hw' as one token or as two letters scores the same
        else:
            out.append(tok)
        i += 1
    return " ".join(out)


for model, res in cap["results"].items():
    hyp = res["c01_status/phone"]["text"]
    print(f"{model:30} WER {wer(normalize(truth), normalize(hyp))[0]:5.0%}   {normalize(hyp)}")
```

All three now score 0% on this call, which is the truth. Normalization is not cosmetic: on Halfwheel's calls it changes which model looks best. Production toolkits ship far more complete normalizers, such as the one in OpenAI's Whisper repository and NVIDIA's NeMo text processing, and the converse step, turning "one oh four two" into "1042" for the database, is called **inverse text normalization**. The receptionist needs that too, and it gets it in Chapter 8.

```{warning}
Normalize references and transcripts with the *same* function, and never tune the normalizer on the model you are evaluating. It is easy to write rules that quietly forgive one model's habits and not another's.
```

## Score the words the business depends on

WER weighs "the" and "Okafor" equally. Halfwheel does not. Every generated call in the manifest lists its **entities**: the ticket, bike model, name and phone number the receptionist must get right. An entity counts as heard if its normalized form appears in the normalized transcript.

```{code-cell} ipython3
calls = [json.loads(l) for l in open("data/calls/manifest.jsonl")]


def entity_hits(call, hyp):
    h = f" {normalize(hyp)} "
    return [(e["type"], f" {normalize(e['written'])} " in h) for e in call["entities"]]


by_id = {c["id"]: c for c in calls}
for model, res in cap["results"].items():
    hits = [hit for key, r in res.items() for hit in entity_hits(by_id[key.split("/")[0]], r["text"])]
    detail = ", ".join(f"{t} {'yes' if ok else 'NO'}" for t, ok in hits)
    print(f"{model:30} {sum(ok for _, ok in hits)}/{len(hits)} entities   {detail}")
```

On Chapter 3's six calls, every model gets the ticket and the phone number and misses the bike and the surname. That is a far more useful summary than any WER, and it is the score I use to choose.

## The test bench

Six calls are an anecdote. For a decision I want enough audio to see patterns, and audio that is not all synthetic:

- **30 Halfwheel calls** (229 seconds), the six from Chapter 3 plus 24 generated from templates with a fixed seed. Nine Kokoro voices, American and British. Between them they contain **48 entities**: 17 bike models, 17 customer names, 9 ticket numbers and 5 phone numbers.
- **80 real recordings** from LibriSpeech {cite}`panayotov2015librispeech`, read English audiobook speech with human transcripts: 40 from the clean test set and 40 from the harder "other" set.
- **Every file twice**: as recorded, and through the same simulated phone line as Chapter 3 (8 kHz, band-limited, mu-law, background noise).

That is 220 files per model. I ran six models on one NVIDIA L4 and saved every transcript. Qwen3-ASR, Whisper and Kyutai ran in fp16, since a free T4 has no bfloat16; the NeMo models ran in their default fp32. Every model fits in a T4's 16 GB. The scoring below runs on those saved transcripts, every time this page is built.

```{code-cell} ipython3
import pandas as pd

cap4 = json.load(open("data/ch4/capture_ch4.json"))
res4 = cap4["results"]
refs = {}
for root, name in [("data/calls", "halfwheel"), ("data/librispeech", "librispeech")]:
    for line in open(f"{root}/manifest.jsonl"):
        row = json.loads(line)
        refs[(name, row["id"])] = row


def score(model, fix=normalize, sets=("halfwheel", "librispeech")):
    """WER per (set, condition), entities heard on phone audio, speed and memory."""
    errs, words = defaultdict(int), defaultdict(int)
    ent = defaultdict(lambda: [0, 0])
    spent = audio = 0.0
    for key, r in res4[model]["files"].items():
        name, cond, uid = key.split("/")
        if name not in sets:
            continue
        ref, hyp = normalize(refs[(name, uid)]["written"]).split(), fix(r["text"]).split()
        errs[name, cond] += sum(edit_ops(ref, hyp).values())
        words[name, cond] += len(ref)
        spent, audio = spent + r["s"], audio + r["audio_s"]
        if name == "halfwheel" and cond == "phone":
            padded = f" {fix(r['text'])} "
            for e in refs[(name, uid)]["entities"]:
                ent[e["type"]][0] += f" {normalize(e['written'])} " in padded
                ent[e["type"]][1] += 1
    out = {f"{n} {c}": errs[n, c] / words[n, c] for n, c in words}
    out.update({t: hit / n for t, (hit, n) in ent.items()})
    out["entities"] = sum(h for h, _ in ent.values()) / sum(n for _, n in ent.values())
    out["x real time"] = audio / spent
    out["peak GB"] = res4[model]["peak_gb"]
    return out


BASE = {"parakeet-ctc-0.6b": "CTC", "parakeet-tdt-0.6b-v2": "TDT", "qwen3-asr-0.6b": "LLM",
        "qwen3-asr-1.7b": "LLM", "whisper-large-v3-turbo": "LLM", "kyutai-stt-2.6b-en": "streaming LM"}
table = pd.DataFrame({m: score(m) for m in BASE if "files" in res4.get(m, {})}).T
table.insert(0, "decoder", [BASE[m] for m in table.index])
pct = ["halfwheel clean", "halfwheel phone", "librispeech clean", "librispeech phone",
       "ticket", "phone", "bike", "name", "entities"]
(table[["decoder"] + pct + ["x real time", "peak GB"]]
 .style.format({c: "{:.1%}" for c in pct} | {"x real time": "{:.0f}x", "peak GB": "{:.1f}"}))
```

Three things stand out.

- **Word error rate barely separates the models.** On Halfwheel's phone calls they span 4.2% to 6.4%. A leaderboard would rank them, but a two-point spread on 229 seconds of audio is not a decision.
- **Entity accuracy is where they differ, and it is low for all of them.** Tickets and phone numbers are fine everywhere. Bike models and names are not: no model gets even half of the bikes right. These are exactly the words the receptionist needs to look anything up.
- **Real speech and phone lines hurt more than synthetic calls.** Halfwheel's synthetic calls lose almost nothing over the phone line, but real LibriSpeech recordings do: Parakeet-CTC goes from 2.6% to 5.1%, and Kyutai from 3.6% to 7.9%. Synthetic test audio flatters every model, which is why I keep real recordings in the bench.

Speed varies far more than accuracy. Parakeet-TDT transcribes 93 times faster than real time; the two Qwen3-ASR models manage about 10x, because their LLM decoder takes one step per token, just as Chapter 3 predicted.

Ignore Kyutai's speed in this table. Kyutai STT is built to stream: it runs alongside the caller in its own server, with a fixed 2.5 second delay. I ran it offline through Hugging Face transformers instead, appended 3 seconds of silence to every file so its delayed output could finish, and PyTorch's compiler gave up on the varying input lengths and fell back to slow step-by-step execution. Its accuracy numbers are fair; its speed number measures my setup, not the model. Chapter 5 runs it the way it was designed to run.

## Teaching the model the shop's words

The cheapest improvement is to tell the model which words to expect. The shop already has the list: its catalogue of bike models and its customer names. The models differ in how they take it:

- **Whisper** accepts a text **prompt**, which it treats as the transcript of earlier audio. I pass a short sentence listing the shop's words.
- **Parakeet** (NeMo) supports **phrase boosting** during decoding: a tree of key phrases that adds a bonus to any hypothesis that follows one of them. The bonus strength, `boosting_tree_alpha`, has to be tuned, so I tried 0.5 and 2.0.
- **Qwen3-ASR**'s `transcribe()` has no hint or context option in the version I used, so it gets no help here.

```python
# NeMo phrase boosting on a pretrained Parakeet model
from omegaconf import open_dict

cfg = model.cfg.decoding
with open_dict(cfg):
    cfg.strategy = "greedy_batch"
    # Pretrained configs have no boosting_tree section, so create it whole.
    cfg.greedy.boosting_tree = {"key_phrases_list": SHOP_WORDS, "context_score": 1.0, "depth_scaling": 2.0}
    cfg.greedy.boosting_tree_alpha = 0.5
model.change_decoding_strategy(cfg)
```

```{code-cell} ipython3
HINTS = ["parakeet-ctc-0.6b", "parakeet-ctc-0.6b + shop words (0.5)", "parakeet-ctc-0.6b + shop words (2.0)",
         "parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v2 + shop words (0.5)", "parakeet-tdt-0.6b-v2 + shop words (2.0)",
         "whisper-large-v3-turbo", "whisper-large-v3-turbo + shop words"]
hints = pd.DataFrame({m: score(m) for m in HINTS}).T
cols = ["halfwheel phone", "librispeech phone", "bike", "name", "phone", "ticket", "entities"]
hints[cols].style.format("{:.1%}")
```

Hints work, and they have a dial you can turn too far.

- **Whisper's prompt is the biggest single win.** Entities go from 65% to 77% and Halfwheel's WER from 4.5% to 3.0%, with no change at all on LibriSpeech.
- **Light boosting on Parakeet-TDT is free.** At 0.5, entities rise from 62% to 71% and real-speech WER does not move.
- **Strong boosting costs accuracy elsewhere.** At 2.0, Parakeet-TDT gets 79% of entities, but its LibriSpeech WER rises from 3.8% to 4.3%. The bonus pushes ordinary speech toward shop words.
- **On CTC, strong boosting backfires.** CTC has no text history to keep the bonus in check. Here is the Daniel Okafor call from Chapter 3 again:

```{code-cell} ipython3
key = "halfwheel/phone/c04_message"
print("said:     ", refs[("halfwheel", "c04_message")]["written"])
for m in ["parakeet-ctc-0.6b", "parakeet-ctc-0.6b + shop words (2.0)"]:
    print(f"{m:38}", res4[m]["files"][key]["text"])
```

## Heard right, spelled wrong: snap to the catalogue

Look at what the models still get wrong, and most of it is not mishearing. "Canyon grizzle" is exactly how *Grizl* is pronounced. "Brompton Sea Line" and "Sophia" sound identical to *C Line* and *Sofia*. No acoustic model can recover a spelling that only exists in the shop's records.

So I do what Chapter 3 did with tickets, for every entity type: compare the transcript against the list the shop already has, and **snap** close matches to the catalogue spelling. Python's standard library is enough. `difflib` scores how similar two strings are between 0 and 1:

```{code-cell} ipython3
import difflib
import functools

CATALOGUE = [  # what the shop knows before the phone rings
    "Trek Domane SL 5", "Specialized Tarmac SL7", "Cannondale Synapse", "Brompton C Line",
    "Giant Defy Advanced 2", "Canyon Grizl", "Surly Cross-Check", "Bianchi Oltre", "Orbea Orca",
    "Ribble Endurance SL", "Daniel Okafor", "Siobhan Murphy", "Tomasz Wojcik", "Mei Nguyen",
    "Raj Patel", "Grace Adeyemi", "Liam O'Sullivan", "Sofia Rossi", "Hannah Schmidt",
    "Kwame Mensah", "Aoife Byrne", "Diego Alvarez"]
CAT = [tuple(normalize(c).split()) for c in CATALOGUE]


@functools.lru_cache(maxsize=None)
def snap(text, cutoff=0.8):
    """Replace stretches of the transcript that closely match a catalogue entry with that entry."""
    words = normalize(text).split()
    candidates = []
    for i in range(len(words)):
        for entry in CAT:
            for n in {len(entry) - 1, len(entry), len(entry) + 1}:  # allow one word split or merged
                if n >= 1 and i + n <= len(words):
                    r = difflib.SequenceMatcher(None, " ".join(words[i:i + n]), " ".join(entry)).ratio()
                    if r >= cutoff:
                        candidates.append((r, i, n, entry))
    owner = [None] * len(words)  # best matches first; never let two matches overlap
    for r, i, n, entry in sorted(candidates, reverse=True):
        if all(o is None for o in owner[i:i + n]):
            owner[i:i + n] = [(i, entry)] * n
    out, k = [], 0
    while k < len(words):
        if owner[k] is None:
            out.append(words[k])
            k += 1
        else:
            start, entry = owner[k]
            out += entry
            while k < len(words) and owner[k] is not None and owner[k][0] == start:
                k += 1
    return " ".join(out)


print(snap("How much would a wheel true cost on a canyon grizzle?"))
print(snap("It's about the Brompton Sea Line."))
print(snap("Hi, it's Sophia Rossi. Is my bike ready?"))
```

```{code-cell} ipython3
def false_snaps(model):
    """Catalogue entries that appear after snapping but were never said."""
    count = 0
    for key, r in res4[model]["files"].items():
        name, cond, uid = key.split("/")
        if name == "halfwheel" and cond == "phone":
            said, heard = f" {normalize(refs[(name, uid)]['written'])} ", f" {snap(r['text'])} "
            count += sum(f" {' '.join(e)} " in heard and f" {' '.join(e)} " not in said for e in CAT)
    return count


rows = {}
for m in list(BASE) + ["parakeet-tdt-0.6b-v2 + shop words (0.5)", "whisper-large-v3-turbo + shop words"]:
    if "files" in res4.get(m, {}):
        before, after = score(m, sets=("halfwheel",)), score(m, fix=snap, sets=("halfwheel",))
        rows[m] = {"entities": before["entities"], "entities, snapped": after["entities"],
                   "WER": before["halfwheel phone"], "WER, snapped": after["halfwheel phone"],
                   "false snaps": false_snaps(m)}
snapped = pd.DataFrame(rows).T
snapped.style.format({c: "{:.1%}" for c in ["entities", "entities, snapped", "WER", "WER, snapped"]} | {"false snaps": "{:.0f}"})
```

The boosted model did not just misspell Daniel Okafor. It replaced him with **Liam O'Sullivan, another customer on the list**, and it started writing the digit "oh" as the letter "o", because one of the boosted phrases begins with "O'". A misspelling is obviously wrong. A different real customer's name looks right, and nobody checks it. That is the worst kind of error for a receptionist.

Snapping is the bigger win. Every base model gains at least 25 points of entity accuracy, and Whisper with its prompt reaches 98%. Word error rate falls too, because the snapped words are real errors fixed, not formatting. And on this set nothing was snapped that the caller did not say.

That last number is flattering, because every caller here is already in the shop's records. A new customer is not:

```{code-cell} ipython3
for new_customer in ["Hannah Smith", "Daniel Okoro", "Raj Patil"]:
    print(f"{new_customer:13} -> {snap(f'Hi, this is {new_customer}, I am a new customer.')}")
```

Three new customers, three existing customers' names. When I tried ten plausible new names, nine were snapped onto someone already in the records. The fix is not a better threshold; it is a conversation. The receptionist should read entities back before acting on them ("That's Hannah Schmidt, with the Cannondale Synapse?"), which is exactly what Chapter 8 builds.

## Halfwheel's choice

For live calls I choose **Parakeet-TDT with light boosting (0.5) and catalogue snapping**: 96% of entities, 4.2% WER on phone calls before snapping, no cost on general speech, and 90 times faster than real time. It also transcribes a finished call in about a tenth of a second, which Chapter 5 turns out to need more than streaming.

Whisper large-v3-turbo with a prompt, plus snapping, scored slightly higher at 98%. It is the better choice when nobody is waiting for an answer, such as transcribing voicemail overnight, where its slower, whole-utterance decoding costs nothing.

## Case study: voicemail in Call Gate

Call Gate, a system I built to keep robocalls out of my voicemail, is that second case. A recorded message arrives, gets transcribed, and a model answers questions about it to decide whether to notify me. The real calls I tested it on were transcribed by Deepgram's hosted nova-3 model. Nobody waits on the line, so the whole pipeline takes 8 to 12 seconds from the end of the message to the notification on my phone, and most of that is the telephony provider delivering the recording, not speech recognition.

Two lessons carried over directly:

- **Batch is fine when nobody is waiting.** You can use the most accurate model and the longest context, because latency only has to beat a human checking their phone.
- **A transcript is untrusted input.** Scam calls say "this is not a sales call" and claim to be your bank. Whatever reads the transcript must treat it as the caller's claim, not as fact. Chapter 9 builds Halfwheel's spam screening on that rule.

## When it breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| Two models' WERs disagree with your users' complaints | WER weighs every word the same | Score the entities your business depends on |
| One model looks far worse than the others | Formatting, not recognition ("1042" vs "one oh four two") | Normalize references and transcripts with the same function |
| Brand names become common words | Rare words lose to frequent ones | Hints (prompt or boosting), then catalogue snapping |
| A boosted model writes a *different* customer's name | Boosting strength too high, especially on CTC | Lower the strength, and measure on general speech too |
| A new customer is matched to an existing one | Snapping assumes everyone is in the records | Read entities back to the caller before acting |
| Great numbers in testing, worse on real calls | Synthetic test audio is too clean | Keep real recordings, and phone-quality versions, in the bench |

```{tip}
**Takeaways**
- Choose a speech model on your own audio and on the words your business depends on, not on leaderboard WER.
- Normalize before you score, or you will be measuring formatting.
- Hints are cheap and effective, but they have a dial. Turn it too far and the model invents plausible words, including real customer names.
- The shop's own records fix more errors than any model swap. Snap to the catalogue, then confirm with the caller.
```

## Try it yourself

The companion notebook, `chapters/04-choosing-asr`, has the full bench, the saved transcripts and the scoring code.

1. **Your own words.** Replace `CATALOGUE` with the product names and customer names of a business you know, generate ten calls with the Chapter 3 generator, and score a model on them.
2. **Tune the cutoff.** Lower `snap`'s cutoff from 0.8 to 0.7. How many more entities does it recover, and how many false snaps appear?
3. **Find the breaking point.** Re-run the Whisper prompt with twice as many shop words. Does it keep helping, or does it start to hurt LibriSpeech?

## Next

Everything in this chapter waited for the caller to finish speaking before transcribing a word. A receptionist cannot do that. In Chapter 5 the receptionist listens *while* the caller talks: streaming recognition, deciding when the caller has finished, and what to do when they interrupt.
