---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
date: "2027-01-04"  # quoted: myst-nb stores front matter as JSON
description: Real robocalls against Halfwheel's genuine callers, including the ones that sound like scams. Keywords, a language model's verdict, typed questions with a learned score, campaign memory, and the rule that matters most, what never to block. With Call Gate as the case study.
---

# Screening calls

The phone rings at Halfwheel Cycles, and the receptionist answers: "Thanks for calling Halfwheel Cycles, how can I help?" A recording replies: "Hello. This call is to inform you that a pre-authorized order has been placed on your name. To authorize this order, press 1." Chapter 8's receptionist is helpful by design. It asks for a ticket number, and the recording carries on talking over it.

Most robocalls are harmless to the receptionist; they waste a minute of an L4's time. The dangerous mistake goes the other way. Blocking the shop's supplier, the bank's fraud team or an angry customer because they *sounded* like a scam costs real money. This chapter screens calls on what the caller says first, the way the receptionist hears it, and measures both kinds of mistake.

Here is where I ended up:

| Finding | Number |
|---|---|
| A word list catches most robocalls, and blocks genuine callers | {glue}`kw_hits` of {glue}`n_unwanted` robocalls caught, and {glue}`kw_false` of {glue}`n_wanted` genuine callers blocked: the supplier, the bank and the angry customer |
| Asking a model "is this spam?" is good but brittle | Qwen3-4B separates the two with AUC {glue:text}`v_auc:.3f`, but waved through {glue}`v_miss` robocalls with near-total confidence |
| Specific questions are as good, and show their working | nine yes/no questions and a small learned score: AUC {glue:text}`q_auc:.3f` on calls it hadn't seen, with weights you can read |
| The policy matters more than the score | a real bank fraud alert looks exactly like a scam; with rules for what never to block, {glue}`pol_false` genuine caller was blocked (a sales rep) and {glue}`pol_block` of {glue}`n_unwanted` robocalls were |
| Robocalls repeat themselves | {glue}`camp_rep` of {glue}`n_robo` real robocalls were near-copies of a script already seen; in Call Gate's first four weeks live, memory decided 44 of 56 calls |
| Screening is cheap | the nine questions take {glue:text}`q_ms:.0f` ms on one L4 |

```{note}
The robocalls are real: 150 English recordings from the NCSU Robocall Audio Dataset {cite}`prasad2023robocall`, collected by the US Federal Trade Commission and released into the public domain. The capture script, `sim/capture_ch9.py`, downloads them, puts them through Chapter 2's phone line and Chapter 4's recogniser, and asks Qwen3-4B the questions on one rented L4. The chapter runs from the saved answers in seconds; the companion notebook reruns the questions on a free Colab T4.
```

## Who rings a bike shop

The bench has two sides. On one side, {glue}`n_robo` real robocalls, plus the 5 synthetic spam calls from Chapter 3. On the other, Halfwheel's genuine callers from Chapters 3, 4 and 8, and twelve *lookalikes*: genuine calls that share words with scams. A supplier chases an overdue invoice. The bank's fraud team asks the shop to call back. An angry customer wants a refund today. An insurer asks for a frame number.

```{code-cell} ipython3
import json
import re
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "sim")
from scoring import normalize  # Chapter 4

heard = json.load(open("data/ch9/heard.json"))  # what Parakeet heard in each call's first 20 s
judged = json.load(open("data/ch9/judged.json"))  # Qwen3-4B's answers
ids = list(heard)
unwanted = np.array([not heard[i]["wanted"] for i in ids])
print(Counter((heard[i]["source"], "unwanted" if not heard[i]["wanted"] else "wanted") for i in ids), "\n")
for i in ["r1098320", "l01", "l03", "l05"]:
    print(f"{i:9s} {heard[i]['category']:15s} {heard[i]['heard'][:120]}")
```

Every call is screened on its first 20 seconds, as heard over Chapter 2's phone line by Chapter 4's recogniser. That's what the receptionist has to work with before it decides how to answer. And as Chapter 4 put it, a transcript is untrusted input: "this is your bank" is the caller's claim, not a fact.

## A list of words

The first thing anyone tries is a list of scam words:

```{code-cell} ipython3
SCAM_WORDS = ["press 1", "press one", "press 2", "press two", "warranty", "social security", "amazon", "irs",
              "lawsuit", "arrest", "suspend", "final notice", "credit card", "interest rate", "medicare", "loan",
              "vehicle", "refund", "account", "debt"]
by_words = np.array([any(w in heard[i]["heard"].lower() for w in SCAM_WORDS) for i in ids])
print(f"robocalls caught: {np.sum(by_words & unwanted)} of {unwanted.sum()}")
print(f"genuine callers blocked: {np.sum(by_words & ~unwanted)} of {(~unwanted).sum()}")
for i in ids:
    if by_words[ids.index(i)] and heard[i]["wanted"]:
        hit = [w for w in SCAM_WORDS if w in heard[i]["heard"].lower()]
        print(f"   {heard[i]['category']:15s} {hit}  {heard[i]['heard'][:80]}")
```

It catches almost every robocall, because robocalls really do say "press 1" and "account". But the supplier talks about the shop's *account*, the bank about the business *account*, and the angry customer wants a *refund*. Each is exactly the kind of call the shop must not miss. A word list can't tell a scam about your account from a genuine call about your account, because the words are the same. What differs is everything around them.

## Asking the model

The receptionist already has a language model, so the obvious next step is to ask it. For each call, I asked Qwen3-4B one question, with the transcript and an instruction to treat it as the caller's claim:

```{code-cell} ipython3
print(judged["system"], "\n\nQuestion:", judged["verdict"])
```

Instead of letting it write an answer, I read its probability of starting with "Yes" against "No", straight from the model's output. That gives a score between 0 and 1 from a single forward pass, with no generated text to parse.

```{code-cell} ipython3
def auc(score, positive):
    """Chance that a random unwanted call scores higher than a random wanted one."""
    pos, neg = score[positive], score[~positive]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


verdict = np.array([judged["calls"][i]["verdict"] for i in ids])
print(f"AUC {auc(verdict, unwanted):.3f}")
for t in [0.5, 0.9]:
    print(f"block when P(yes) >= {t}: robocalls caught {np.sum((verdict >= t) & unwanted)} of {unwanted.sum()}, "
          f"genuine callers blocked {np.sum((verdict >= t) & ~unwanted)} of {(~unwanted).sum()}")
print("\nrobocalls the model was surest were fine:")
for i in sorted([i for i in ids if not heard[i]["wanted"]], key=lambda i: judged["calls"][i]["verdict"])[:6]:
    print(f"   P(yes)={judged['calls'][i]['verdict']:.3f}  {heard[i]['heard'][:110]}")
print("\ngenuine callers it was surest were spam:")
for i in sorted([i for i in ids if heard[i]["wanted"]], key=lambda i: -judged["calls"][i]["verdict"])[:2]:
    print(f"   P(yes)={judged['calls'][i]['verdict']:.3f}  {heard[i]['category']}: {heard[i]['heard'][:90]}")
```

The model is good. It blocks one genuine call, not three, and separates the two groups far better than the word list. But look at what it let through, with near-total confidence: a "pre-authorized order" scam and a "press 1 for English" menu. And the genuine call it blocked was the bank's fraud alert, the one call a shop most needs to hear about. A single yes/no question asks the model to do everything at once: notice the cues, weigh them, and decide what counts as unwanted for a bike shop. When it gets that wrong, there's nothing to inspect.

## Asking better questions

Call Gate, the robocall screen I built for my own phone (its case study is below), asks a different kind of question. Instead of one verdict, it asks a model several narrow yes/no questions about the evidence, and lets a small learned score combine the answers. I adapted its questions for a business line:

```{code-cell} ipython3
QUESTIONS = judged["questions"]
X = np.array([[judged["calls"][i]["answers"][q] for q in QUESTIONS] for i in ids])
print(f"{'question':12s} {'robocalls':>10s} {'genuine':>8s}   (average P(yes))")
for k, q in enumerate(QUESTIONS):
    print(f"{q:12s} {X[unwanted, k].mean():10.2f} {X[~unwanted, k].mean():8.2f}   {QUESTIONS[q][:70]}")
```

Most of the questions split the two groups cleanly. Robocalls *sound recorded* and ask you to *press a key*; genuine callers mention *something specific to the shop* and are *about bikes*. One question is useless: robocalls and genuine callers insist they're legitimate equally often. And "asks for personal or payment details" almost never fires, because in the first twenty seconds a robocall asks you to press 1, not to read out your card. Those are findings about the questions, and you only get them by asking narrow ones.

To combine the answers, I fit a logistic regression: one weight per question and a bias, learned from labelled calls. To score each call fairly, I train on four fifths of the bench and score the other fifth, five times over, so no call is ever scored by weights that saw it.

```{code-cell} ipython3
def fit(X, y, l2=0.1, steps=3000, lr=0.5):
    """Logistic regression by gradient descent, on log-odds features."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
        b -= lr * np.mean(p - y)
    return w, b


logodds = np.log(np.clip(X, 1e-4, 1 - 1e-4) / np.clip(1 - X, 1e-4, 1))
folds = np.random.default_rng(0).permutation(len(ids)) % 5
score = np.zeros(len(ids))
for f in range(5):
    w, b = fit(logodds[folds != f], unwanted[folds != f].astype(float))
    score[folds == f] = 1 / (1 + np.exp(-(logodds[folds == f] @ w + b)))
w, b = fit(logodds, unwanted.astype(float))  # weights on the whole bench, to read them
print(f"AUC on unseen calls {auc(score, unwanted):.3f}\n")
for q, wq in sorted(zip(QUESTIONS, w), key=lambda t: -abs(t[1])):
    print(f"{q:12s} {wq:+.2f}")
for t in [0.5, 0.9]:
    print(f"\nblock when score >= {t}: robocalls caught {np.sum((score >= t) & unwanted)} of {unwanted.sum()}, "
          f"genuine callers blocked {np.sum((score >= t) & ~unwanted)} of {(~unwanted).sum()}", end="")
```

The weights are readable, which a single verdict never is: the score leans on "asks you to press a key", "sounds recorded" and "claims to be an institution", and against "about bikes". On raw separation, the questions with a learned score did about as well as the single verdict, not better. At a threshold of 0.9 they caught more robocalls but blocked one more genuine caller; at 0.5, four more. What the questions buy is not a better number. It's evidence you can read, and handles for the rules that come next.

## Robocalls repeat themselves

Scam campaigns reuse their scripts. The same "Ms. Morgan, about your supply charges" message appears in the dataset again and again. That gives a second, almost free signal: remember what you've already seen. Call Gate compares each new message against past ones by word *shingles*, runs of three consecutive words with digits replaced, and counts a match when the two sets overlap by 60% or more:

```{code-cell} ipython3
def shingles(text):
    words = re.sub(r"\d", "#", normalize(text)).split()
    return {tuple(words[k:k + 3]) for k in range(len(words) - 2)}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


order = list(np.random.default_rng(1).permutation(ids))  # every call, in some order of arrival
known_scams, repeats = [], {}
for i in order:
    s = shingles(heard[i]["heard"])
    repeats[i] = any(jaccard(s, t) >= 0.6 for t in known_scams)
    if not heard[i]["wanted"]:
        known_scams.append(s)  # a person labels it a scam once they've heard it
robo = [i for i in ids if heard[i]["source"] == "NCSU robocall"]
print(f"robocalls matching a script already labelled as a scam: {sum(repeats[i] for i in robo)} of {len(robo)}")
print(f"genuine callers matching one: {sum(repeats[i] for i in ids if heard[i]['wanted'])}")
```

Nearly half of these real robocalls were near-copies of one that had already come in, and no genuine caller matched any of them. Campaign memory only works from the second sighting, and only once someone has labelled the first. But for a known script, it's exact: no model has to weigh anything.

## What never to block

A score is not a decision. The decision needs a policy, and the policy has to reflect the costs. Letting a robocall through costs a minute. Blocking the shop's bank, or a customer, can cost far more. Look at how the questions saw the bank's genuine fraud alert:

```{code-cell} ipython3
k = ids.index("l03")
print(heard["l03"]["heard"], "\n")
print({q: round(float(X[k, j]), 2) for j, q in enumerate(QUESTIONS)})
```

On the evidence, it's a robocall. It's an automated message, it names an institution, and it asks the listener to call a number. Real banks really do call like this. No question about the words can tell this call from the "pre-authorized order" scam, so no score built on them can either. The answer is a rule: a call that claims to be a bank, a government or a large company is never blocked on its score. It goes to a person as a message, flagged as a possible impersonation, and the person checks it by calling the institution back on a number they already know.

So the shop's own business comes first, the memory of known scams next, then claims to be an institution, and the score decides only what's left:

```{code-cell} ipython3
A = {q: X[:, j] for j, q in enumerate(QUESTIONS)}


def decide(k):
    i = ids[k]
    if A["relevant"][k] >= 0.5 or A["specific"][k] >= 0.5:
        return "answer"  # about the shop or its business: never block
    if repeats[i]:
        return "block"  # a script already labelled as a scam
    if A["institution"][k] >= 0.5:
        return "take a message"  # a person decides, with an impersonation warning
    if score[k] >= 0.9:
        return "block"
    return "take a message" if score[k] >= 0.5 else "answer"


decision = np.array([decide(k) for k in range(len(ids))])
for side, mask in [("robocalls", unwanted), ("genuine callers", ~unwanted)]:
    print(f"{side:16s}", {d: int(np.sum(decision[mask] == d)) for d in ["block", "take a message", "answer"]})
print("\ngenuine callers that didn't get a normal answer:")
for k, i in enumerate(ids):
    if heard[i]["wanted"] and decision[k] != "answer":
        print(f"   {decision[k]:15s} {heard[i]['category']:15s} {heard[i]['heard'][:80]}")
```

```{code-cell} ipython3
:tags: [remove-cell]
from myst_nb import glue

glue("n_unwanted", int(unwanted.sum()), display=False)
glue("n_wanted", int((~unwanted).sum()), display=False)
glue("n_robo", len(robo), display=False)
glue("kw_hits", int(np.sum(by_words & unwanted)), display=False)
glue("kw_false", int(np.sum(by_words & ~unwanted)), display=False)
glue("v_auc", auc(verdict, unwanted), display=False)
glue("v_miss", int(np.sum((verdict < 0.01) & unwanted)), display=False)
glue("q_auc", auc(score, unwanted), display=False)
glue("pol_block", int(np.sum((decision == "block") & unwanted)), display=False)
glue("pol_msg", int(np.sum((decision == "take a message") & unwanted)), display=False)
glue("pol_false", int(np.sum((decision == "block") & ~unwanted)), display=False)
glue("camp_rep", int(sum(repeats[i] for i in robo)), display=False)
glue("q_ms", 1000 * float(np.median([judged["calls"][i]["questions_s"] for i in ids])), display=False)
```

Three outcomes, not two. Calls about the shop's business are always answered. Known scam scripts are always blocked. Calls naming an institution, and calls the score isn't sure about, become messages for a person, who can delete a junk message in two seconds. A blocked customer never calls back. The cost of this caution is the {glue}`pol_msg` robocalls that end up as messages rather than being blocked outright. Four genuine callers become messages too: the bank, the council, the landlord and, less happily, a customer asking where the shop is and where to park, whom the questions didn't recognise as being about the shop. That's a price a small shop can pay, and it falls as campaign memory learns. A handful of robocalls get a normal answer, because the questions took them for genuine callers; Chapter 8's receptionist then spends a minute on them and hangs up when they don't answer its questions. The one genuine call still blocked is the saddle rep's sales visit. My bench counts it as wanted, the score doesn't, and a shop could reasonably side with either.

## Case study: Call Gate

Call Gate screens robocalls to my own phone number, and it has run every day since I built it. It solves a narrower problem than Halfwheel's receptionist, keeping recorded scam messages out of my voicemail, but every idea in this chapter comes from it:

- **Gate the risky calls only.** Only calls from toll-free numbers, unless they're on an allow list, go to the screen; everything else rings through untouched.
- **A challenge a recording can't pass.** A gated caller hears "To leave a message, press" and a random digit. A person presses it; a recording plays on. That one step separates people from robots before any model runs.
- **Typed questions and a learned score.** The message is transcribed (Deepgram's nova-3) and a model answers about nine questions in one call: pressure, a push to press a key or give details, a claim to be an institution, specific details, a sales pitch. A logistic score combines them. The model, TypeSafe's Jev, answers typed questions with calibrated probabilities and cannot write free text. That's the property the Qwen3 trick above imitates.
- **Rules before the score.** Anything claiming to be financial, healthcare or education is always delivered, with an impersonation warning when the evidence looks like a scam.
- **Campaign memory**, as above, with my own taps on "Scam" or "Not scam" labelling campaigns.
- **Shadow mode first.** New rules start by recording what they *would* have done while delivering everything, and are switched on only when the record looks right.

In testing, Call Gate held back none of 130 genuine-style alerts, kept 5 of 5 real toll-free scam calls out of voicemail in its first two days, and took 8 to 12 seconds from the end of a message to the notification on my phone, almost all of it the telephony provider delivering the recording. The questions themselves took about 0.2 s, at around $0.00003 a call. The genuine alerts in that test were written by a language model, so those are lab results, not guarantees. That's why it went live in shadow mode.

### Call Gate in production

Over its first four weeks live, from October 16 to November 11, 2026, Call Gate screened 56 calls. All of them came from the database it keeps, counted by query, with no caller details:

| | Calls |
|---|---|
| Gated calls, all from robots (no person who passed the digit challenge left a message) | 56 |
| Dead air, ignored | 2 |
| Decided by campaign memory: a script I'd already labelled as a scam | 44 |
| Matched a junk campaign | 1 |
| Delivered under the protected-sector rule (bank, lender, healthcare, education) | 5 |
| Score unsure, delivered for review | 4 |
| Would have been held back, if enforcement had been on | 45 |
| Actually held back | 0 (shadow mode the whole time) |

The digit challenge did exactly what it was built for. From Twilio's own call logs: after go-live, 55 calls from toll-free numbers reached the challenge, and not one pressed a key, right or wrong. Fifty-four stayed silent through it (52 of them leaving a recorded message; the table above also counts 4 messages from the days just before go-live) and one hung up. Over the same weeks, 97 calls from ordinary numbers rang straight through to my phone without meeting the gate at all, so the screen only ever touched about a third of my calls. A recording can't press a digit it wasn't told in advance, and a person who rings from an ordinary number never has to.

Three more things stand out, and all three echo this chapter's bench.

**Memory did most of the work.** On the bench, campaign memory matched about half the robocalls. In production it decided 44 of 56 calls, because scam campaigns repeat even more in real life: the largest campaign reached my phone 30 times, from 30 different numbers. Blocking numbers would have caught none of those repeats; remembering scripts caught all of them.

**The misses were in the rule that never blocks.** Of the calls Call Gate would have delivered, 8 turned out to be scams. Four were the first calls of a loan-approval robocall on launch day, before I'd labelled its script, which is how memory is meant to start. The other four were reworded versions of the same loan scam, which the model judged to come from a lender, a protected sector. The rule that keeps real banks from being blocked let them through, two with an impersonation warning and two with none. It's the bank fraud alert from the bench, seen from the other side: a scam that claims to be a lender looks like a lender.

**Shadow mode found it first.** Nothing was ever held back. Every decision was recorded as what it *would* have done, and I labelled calls as they arrived: 38 as scams, none as genuine. None of the calls it would have held has turned out to be genuine, though 15 of them are still unlabelled, so that isn't proven yet. And the protected-sector gap showed up before any enforcement was switched on, which is exactly when you want to find it. The analysis itself took a median of 1.8 s per call (90th percentile 3.1 s), from the start of transcription to the decision.

What Call Gate doesn't have to handle, Halfwheel does. A voicemail screen can wait a few seconds and decide on the whole message. A receptionist has to decide while the caller is still on the line, on the first sentence or two. And it can't ask every customer to press a random digit before they speak.

## Halfwheel's screen

Putting the chapter together, the receptionist screens like this:

1. Answer as normal. Screening runs on the caller's first turn, the same transcript the receptionist already has.
2. Ask the nine questions in one batch, about half a second on a GPU, while the language model writes its reply.
3. About the shop or its business: answer, however it sounds.
4. Known scam scripts, matched by campaign memory: hang up straight away.
5. Claims to be a bank, a government or a big company: "I'll take a message and someone will call you back," flagged for a person.
6. A high score: a polite "Sorry, we don't take these calls" and hang up. In between: take a message.
7. Run all of it in shadow mode first, logging what it would have done, and switch it on only when the log looks right.

## When it breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| Suppliers or customers get blocked | Word lists, or a score with no rules on top | Never block calls about the shop's business; check the lookalikes first |
| A bank's or insurer's call never reaches anyone | Institutions treated as spam | Send institution claims to a person, flagged, never to the block list |
| Obvious scams get through | A single yes/no verdict missing the cues | Ask narrow questions; combine them with a learned score |
| The model believes "this is not a sales call" | The transcript treated as fact | Treat transcripts as claims; self-assertion is not evidence either way |
| A new scam wave gets through | No memory of past calls | Campaign memory on labelled scripts |
| A rule change blocks good calls in production | Switched on without evidence | Shadow mode: log what it would do, then enforce |

```{tip}
**Takeaways**
- Screening has two kinds of mistake, and they don't cost the same. Decide what must never be blocked before you decide what to block.
- Word lists catch robocalls and block the customers who use the same words.
- Narrow yes/no questions, answered as probabilities and combined by a small learned score, beat one big verdict, and they show you why.
- Institutions are the hardest case: the real ones matter most, and scams imitate them. Send them to a person.
- Scams repeat. A memory of labelled scripts is cheap and precise.
- Ship a screen in shadow mode first.
```

## Try it yourself

The companion notebook, `chapters/09-screening-calls`, has the saved answers, every cell above and the capture script.

1. **Your own lookalikes.** Add three genuine calls your own business gets that sound like scams. Does the policy still let them through?
2. **Drop a question.** Remove `relevant` and `specific` from the score and the rules. What happens to the genuine callers?
3. **Tune the threshold.** Plot robocalls caught against genuine callers blocked as the block threshold moves from 0.5 to 0.99.
4. **Memory from day one.** Replay the robocalls in date order with campaign memory only. How many would a person have had to label to stop the rest?

## Next

The receptionist now listens, hears, thinks, speaks and screens, as a pipeline of separate models with text in between. Chapter 10 looks at models that skip the text: Spirit LM, Moshi and CSM, which take speech in and give speech out, and what that buys in timing and tone of voice.
