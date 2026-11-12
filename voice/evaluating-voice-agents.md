---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
date: "2027-01-18"  # quoted: myst-nb stores front matter as JSON
description: How much a voice agent's score is worth. The receptionist under new voices, a noisier line and repeated runs; rules against a model as judge; and what Call Gate and Kadence show about the gap between a bench and real calls.
---

# Evaluating voice agents

Chapter 8 ended with a number: Halfwheel's receptionist handled 11 of 16 calls. It's a satisfying number to put in a table. It's also one bench, one set of voices, one phone line, one run, and one judge that I wrote myself. Change any of those, and the number moves.

This chapter asks what a voice agent's score is worth. I rerun Chapter 8's receptionist on the same calls with different caller voices and a noisier line, run it five more times with sampling switched on, and have a separate model grade every conversation. Then I look at the two systems in this book that have met real callers, Call Gate and Kadence, to see what no bench caught.

Here is where I ended up:

| Finding | Number |
|---|---|
| Sixteen calls is a wide interval | 11 of 16 handled is anywhere from {glue:text}`ci_lo:.0f`% to {glue:text}`ci_hi:.0f`% at 95% confidence |
| The total hides which calls fail | the score stayed between {glue}`cond_min` and {glue}`cond_max` of 16 across voices and noise, but {glue}`flips` of 16 calls changed outcome somewhere |
| Runs vary even with the same audio | five sampled reruns scored {glue}`samp_min` to {glue}`samp_max` of 16 |
| A model judge with a vague question is worse than none | asked "did the caller get what they needed?", it agreed with the rules on {glue:text}`j1_agree:.0f`% of {glue}`n_judged` conversations, below chance (kappa {glue:text}`j1_kappa:.2f`) |
| Told the expected behaviour, it's useful | agreement {glue:text}`j2_agree:.0f`% (kappa {glue:text}`j2_kappa:.2f`), and it caught a made-up address the rules missed in {glue}`j2_inv` conversations |
| The field finds what the bench can't | Call Gate's real misses and Kadence's café failures were all outside their test sets |

```{note}
The new runs come from `sim/capture_ch11.py`, which ran Kokoro, Parakeet and two Qwen3 models on one rented L4 in about an hour. On a free Colab T4 the receptionist reruns, but the judge, Qwen3-8B, needs a larger GPU. Everything else is saved data from earlier chapters, so this chapter runs on a CPU in seconds.
```

## What to measure

Every chapter in this book built a bench for one layer of the receptionist. Put together, they're an evaluation plan:

| Layer | What can go wrong | Measured as | Chapter | Halfwheel's number |
|---|---|---|---|---|
| Audio | wrong sample rate, clipping, a bad phone line | WER and entities under each mistake | 2 | the wrong sample rate took WER from 5% to 46% |
| Hearing | a word or a name misheard | WER, and entity accuracy | 4 | 96% of entities with boosting and snapping |
| Turn-taking | cutting callers off, or making them wait | cut-offs and wait after the turn ends | 5 | 3 of 16 cut off at 1.2 s, 1.2 s wait |
| Speaking | the wrong words, or the right words unclear | facts heard back through ASR | 6 | 65 of 66 facts after normalisation |
| Delivery | the wrong tone | pitch, range, pace; listening | 7 | the reference clip moved pitch by 33 Hz, the tag by less |
| The task | wrong actions, invented facts | handled calls, wrong actions | 8 | 11 of 16 handled, 1 wrong action (with the guard in code) |
| Screening | blocking a customer | false blocks and missed robocalls | 9 | 1 genuine caller blocked, 100 of 155 robocalls blocked |
| Timing | the caller waits | seconds from their last word to the reply | 8 | 2.4 s median |

Each row has its own bench because each layer fails differently, and a single end-to-end score can't say which layer broke. But the end-to-end score is still the one that matters to the caller, so the rest of this chapter takes Chapter 8's "11 of 16" and tests how much it can be trusted.

## Sixteen calls is a wide interval

A score from 16 calls has a lot of luck in it. The usual way to put a range on a proportion is the Wilson interval:

```{code-cell} ipython3
import glob
import json
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "sim")
src = open("building-the-receptionist.md").read()  # Chapter 8's judge, read from that chapter so they can't drift
fence = "`" * 3  # written this way so this cell can contain it
cells = re.findall(fence + r"\{code-cell\} ipython3\n(.*?)" + fence, src, re.S)
ns = {}
exec(cells[0].split("print(json.dumps")[0], ns)
exec([c for c in cells if "def judge(" in c][0].split("RUNS = [")[0], ns)
judge, BY_ID = ns["judge"], ns["BY_ID"]


def wilson(k, n, z=1.96):
    """95% interval for a proportion k/n that behaves well for small n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


for k, n in [(11, 16), (110, 160), (1100, 1600)]:
    lo, hi = wilson(k, n)
    print(f"{k:5d} of {n:5d} handled: {k / n:.0%}, 95% interval {lo:.0%} to {hi:.0%}")
```

Eleven of sixteen is 69%, but the data are just as consistent with a receptionist that handles 44% of calls as one that handles 86%. You'd need ten times as many calls to narrow that to about ±7 points. That's the first reason to grow a bench: not to find new failures, but to know whether a change made things better at all.

## The same calls, different callers

To grow the bench cheaply, I kept Chapter 8's sixteen scripts and changed who says them and how they sound. Kokoro read every caller line again with a different voice (another speaker with the same accent), and Chapter 2's phone line added noise at 10 dB below the speech instead of 20. That gives three new conditions, plus the original:

```{code-cell} ipython3
def calls(path):
    return json.load(open(path))["calls"]


conditions = {c: calls(f"data/ch11/talk_{c}.json") for c in ["base", "noisy", "voices", "both"]}
scores = {c: {cid: judge(BY_ID[cid], t) for cid, t in log.items()} for c, log in conditions.items()}
print("condition  handled  wrong actions")
for c, s in scores.items():
    print(f"{c:10s} {sum(x['handled'] for x in s.values()):4d}/16 {sum(x['wrong'] for x in s.values()):10d}")
cids = sorted(BY_ID)
print("\n      base noisy voices both")
for cid in cids:
    marks = ["  ok " if scores[c][cid]["handled"] else "  -- " for c in scores]
    if len(set(marks)) > 1:
        print(cid, "".join(marks), BY_ID[cid]["kind"])
pooled = sum(x["handled"] for s in scores.values() for x in s.values())
lo, hi = wilson(pooled, 64)
print(f"\npooled: {pooled} of 64, 95% interval {lo:.0%} to {hi:.0%}")
```

The total barely moved, from 9 to 11 of 16. Underneath it, individual calls flipped. Siobhan Murphy's booking, which failed in Chapter 8 because Parakeet heard "Javon", succeeded with a different caller voice that Parakeet heard correctly. Mei Nguyen's booking went the other way: it succeeded with the original voice and failed in all three new conditions. The noisier line broke a phone-number message. If I had only looked at totals, I'd have concluded that voices and noise don't matter. They do; they just move the failures around.

That's the case for keeping per-call results, not just a score: a "no change" in the total can hide a regression on the call that matters most to the business.

## The same calls, run again

Chapter 8 used greedy decoding, which always picks the most likely next word, so the same audio always gives the same conversation. Many production systems sample instead, which makes replies less repetitive. To see what that costs, I reran the original condition five times with sampling (temperature 0.7):

```{code-cell} ipython3
sampled = {s: {cid: judge(BY_ID[cid], t) for cid, t in calls(f"data/ch11/talk_base-sampled-{s}.json").items()}
           for s in range(5)}
totals = [sum(x["handled"] for x in r.values()) for r in sampled.values()]
print("handled in five sampled runs:", totals)
print("\ncalls that changed between runs:")
for cid in cids:
    marks = "".join("ok " if sampled[s][cid]["handled"] else "-- " for s in sampled)
    if len(set(marks.split())) > 1:
        print(f"   {cid}  {marks}  {BY_ID[cid]['kind']}")
```

Same audio, same receptionist, and the score moved by a call. A change you make to a prompt has to beat that noise before you can believe it helped. Either evaluate with greedy decoding, so that runs are repeatable, or run each version several times and compare the spread, not a single number.

## Who decides "handled"?

Everything so far used Chapter 8's judge: rules I wrote for each call, checking tool calls, their arguments and a few patterns in the replies. Rules are exact, but someone has to write them for every call, and they only check what they were told to. The popular alternative is to ask a language model to grade the conversation. I tried it with Qwen3-8B, a different and larger model than the receptionist's, on all {glue}`n_judged` conversations from Chapters 8 and 11.

First, with the question most people would write: here is the caller's goal and the conversation; did the caller get what they needed?

```{code-cell} ipython3
def kappa(a, b):
    """Agreement beyond chance between two yes/no raters."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe)


judge1 = json.load(open("data/ch11/llm_judge.json"))
judge2 = json.load(open("data/ch11/llm_judge2.json"))
rows = []
for f in sorted(glob.glob("data/ch8/talk_*.json")) + sorted(glob.glob("data/ch11/talk_*.json")):
    run = ("ch8/" if "/ch8/" in f else "ch11/") + f.split("talk_")[1][:-5]
    for cid, turns in calls(f).items():
        rule = judge(BY_ID[cid], turns)
        rows.append({"call": cid, "rules": rule["handled"], "rules_invented": rule["invented"],
                     "judge1": judge1["calls"][f"{run}/{cid}"]["done"] > 0.5,
                     "judge2": judge2["calls"][f"{run}/{cid}"]["correct"] > 0.5,
                     "judge2_invented": judge2["calls"][f"{run}/{cid}"]["invented"] > 0.5})
rules = [r["rules"] for r in rows]
for name in ["judge1", "judge2"]:
    j = [r[name] for r in rows]
    print(f"{name}: agrees with the rules on {np.mean(np.array(j) == np.array(rules)):.0%} of {len(rows)}, "
          f"kappa {kappa(j, rules):+.2f}, says handled {np.mean(j):.0%} (rules: {np.mean(rules):.0%})")
print("\nfirst judge, where it disagreed with the rules:")
print("   rules say handled, judge says not:", Counter(r["call"] for r in rows if r["rules"] and not r["judge1"]).most_common(4))
print("   rules say failed, judge says handled:", Counter(r["call"] for r in rows if not r["rules"] and r["judge1"]).most_common(5))
```

The first judge agreed with the rules less often than chance would predict. Its disagreements tell you why:

- **It failed the calls where the right answer is "no".** The caller asking for a pizza place (c10), an e-bike motor repair (c11), or how long a wheel true takes (c07) didn't get what they wanted, because the shop can't give it. The receptionist was right to decline; the judge, asked whether the caller got what they needed, said no.
- **It passed calls with exactly the mistakes that matter.** A booking under "Aoifeyan" instead of Aoife Byrne (c04), a Sunday caller never told the shop is closed (c05), an invented address (c15): the judge saw a friendly, finished conversation and said yes.

So I asked again, the way the rules are written: here is what a correct receptionist does, with the exact details ("Book a tune-up on Thursday under exactly the name Aoife Byrne. Any other spelling of the name is wrong."), and here is the conversation; did it do that?

```{code-cell} ipython3
print("second judge, where it disagreed with the rules:")
print("   rules say handled, judge says not:", Counter(r["call"] for r in rows if r["rules"] and not r["judge2"]).most_common(4))
print("   rules say failed, judge says handled:", Counter(r["call"] for r in rows if not r["rules"] and r["judge2"]).most_common(5))
print("\nmade-up facts flagged: rules", sum(r["rules_invented"] for r in rows), ", second judge",
      sum(r["judge2_invented"] for r in rows), Counter(r["call"] for r in rows if r["judge2_invented"]).most_common(3))
```

```{code-cell} ipython3
:tags: [remove-cell]
from myst_nb import glue

lo, hi = wilson(11, 16)
glue("ci_lo", float(100 * lo), display=False)
glue("ci_hi", float(100 * hi), display=False)
cond_totals = [sum(x["handled"] for x in s.values()) for s in scores.values()]
glue("cond_min", int(min(cond_totals)), display=False)
glue("cond_max", int(max(cond_totals)), display=False)
glue("flips", int(sum(len({scores[c][cid]["handled"] for c in scores}) > 1 for cid in cids)), display=False)
glue("samp_min", int(min(totals)), display=False)
glue("samp_max", int(max(totals)), display=False)
glue("n_judged", len(rows), display=False)
glue("j1_agree", 100 * float(np.mean(np.array([r["judge1"] for r in rows]) == np.array(rules))), display=False)
glue("j1_kappa", float(kappa([r["judge1"] for r in rows], rules)), display=False)
glue("j2_agree", 100 * float(np.mean(np.array([r["judge2"] for r in rows]) == np.array(rules))), display=False)
glue("j2_kappa", float(kappa([r["judge2"] for r in rows], rules)), display=False)
glue("j2_inv", int(sum(r["judge2_invented"] and r["call"] == "c15" for r in rows)), display=False)
```

With the expected behaviour in front of it, the same model agreed with the rules on most conversations, and it found something the rules had missed. My rules for the parking-and-address call checked that "Mill Lane" was mentioned, so an invented "123 Main Street" counted as a missing fact, not as made up. The second judge flagged it as invented, which is the more useful label. The remaining disagreements are of two kinds. Some are where my own specification is vague. For the wheel-true question, the rules only check that no repair time was invented; the judge, told the receptionist should *say* it has no information, marks silence on the point as a failure. Both are defensible, and the specification needs a sentence. The others are where the judge is still more lenient than the rules. It passed Sunday callers who were booked for Monday without ever being told the shop is closed (c05), and unknown-ticket calls where the message wasn't taken with the right details (c03). Those are checkable facts, which is the argument for leaving them to the rules.

Three rules for model judges follow:

1. **Give it the expected behaviour, not the caller's goal.** A judge that doesn't know the right answer grades friendliness.
2. **Check it against labelled examples before trusting it.** I could only see the first judge was useless because I had the rules to compare with. A few dozen hand-labelled calls are enough to find out.
3. **Use rules for what can be checked exactly.** The booking's name, day and service are fields in a tool call; compare them in code. Use the model judge for what code can't see, like a made-up address or a rude reply.

## The lab and the field

Benches find the failures you thought to test. The two systems in this book that met real callers both failed in ways their benches didn't include.

**Call Gate**, in testing, held back none of 130 genuine-style alerts and separated scams from genuine calls almost perfectly. In its first four weeks live (Chapter 9), the misses that mattered were four reworded loan scams that the model judged to come from a lender. They were let through by the rule that keeps real banks from being blocked. No test call was a scam pretending to be a lender in words the campaign memory hadn't seen. Shadow mode found it, because every live decision was logged and I labelled the calls as they arrived.

**Kadence**, my stand-up meeting agent (Chapter 10), scored 91.6% on 155 generated scenarios after prompt hardening. In real meetings, background chatter in a café triggered unintended comments on a real ticket, noise turned "COG-13" into "COG 31", and a missing workflow state produced an error instead of a reply. None of those were in the generated scenarios, because the generator had never been in a café.

Both point to the same practice: treat the bench as the floor, and keep evaluating after launch.

- **Log every call** with what was heard, what the model did, and what each tool returned. That's the raw material for everything else.
- **Shadow new behaviour first.** Run the new version alongside the old one, record what it would have done, and compare.
- **Label a sample of real calls every week,** and add the interesting failures to the bench, so the bench grows towards reality.
- **Watch the per-layer numbers in production:** entity accuracy, cut-offs, wait times, wrong actions. A drop in one tells you where to look.

## Halfwheel's evaluation plan

Putting the chapter together, Halfwheel's receptionist is evaluated like this:

1. **Per layer, on every change:** the benches from Chapters 2 to 9, each with its own number.
2. **End to end, on every change:** Chapter 8's calls under all four conditions, greedy decoding, with per-call results compared against the last version, not just the total.
3. **Judged by rules where possible,** with a model judge given the expected behaviour for the rest, checked against a labelled sample.
4. **After launch:** every call logged, changes shadowed first, and a weekly sample of real calls labelled and added to the bench.

## When it breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| A change "helps" by one call | Noise: 16 calls, or sampling | Grow the bench; use greedy decoding or several runs; look at intervals |
| The total is stable but customers complain | Failures moved between calls | Compare per-call results between versions |
| A model judge rates everything highly | It's grading friendliness, not correctness | Give it the expected behaviour and exact details; check it against labels |
| A model judge fails correct refusals | It was asked about the caller's goal | Ask about the expected behaviour instead |
| The agent fails in production in ways never tested | The bench came from your imagination | Log real calls, label a sample, and grow the bench from them |

```{tip}
**Takeaways**
- A score from a small bench is a wide interval. Report it as one, and grow the bench before trusting small differences.
- Change the voices and the line, not just the scripts. Totals hide which calls flip.
- Sampled decoding moves the score between identical runs. Evaluate repeatably, or evaluate several times.
- A model judge is only as good as its question. Tell it the expected behaviour, and check it against labelled examples.
- Benches find the failures you imagined. Logging, shadow mode and labelled real calls find the rest.
```

## Try it yourself

The companion notebook, `chapters/11-evaluating-voice-agents`, has the saved runs, every cell above and the capture script.

1. **How many calls?** Using the Wilson interval, how many calls would you need to tell 70% from 80% handled apart?
2. **A third condition.** Add a condition with a faster speaking rate (Kokoro's `speed=1.3`) for every caller. Which calls flip?
3. **Judge the judge.** Label twenty of the conversations yourself. Which judge agrees with you more, the rules or the second model judge?
4. **A judge for tone.** Write a model-judge question for "the receptionist was polite and didn't interrupt" and check it on ten calls you've listened to.

## Next

The receptionist has been heard, timed, voiced, screened and evaluated, all on recordings and simulated phone lines. Chapter 12 connects it to a real phone number and looks at what changes when the line, the caller and the clock are all real: telephony, streaming audio both ways, scaling to many calls at once, and what it costs.
