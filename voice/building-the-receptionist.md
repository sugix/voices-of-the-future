---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
date: "2026-12-28"  # quoted: myst-nb stores front matter as JSON
description: Chapters 2 to 7 joined into one receptionist with a small language model and four tools, run against sixteen callers who talk back. What breaks, what fixes it, and where the waiting goes.
---

# Building the receptionist

Aoife Byrne rings Halfwheel Cycles to book a tune-up for her Trek on Thursday. The receptionist asks her name, and she says it. The phone line and Parakeet turn "Aoife Byrne" into "Epha Bion", and the receptionist, doing exactly what it was told, reads it back: "So, Epha Bion, for a tune-up on Thursday?" Aoife corrects it and spells her name. The correction goes through the same phone line and the same recogniser, and comes out as "Aoifeyan, A-O-I-S-E". She gives up and says yes. A tune-up for "Aoifeyan" goes into the shop's diary.

Every piece of that receptionist works. Chapter 5's listening, Chapter 4's recogniser and Chapter 6's voice each passed their own tests. This chapter joins them to a small language model with four tools, and puts the whole thing in front of sixteen callers who talk back. Then I measure two things: did the caller get what they rang for, and how long did they wait each time they stopped talking?

Here is where I ended up:

| Finding | Number |
|---|---|
| A 4B model is the smallest that works | Qwen3 1.7B handled {glue}`small_ok` of 16 calls; 4B handled {glue}`main_ok`; 8B handled {glue}`big_ok`, no better than 4B |
| Mishearing costs calls, mostly through names | with perfect transcripts the 4B model handled {glue}`text_ok` calls; through the phone and Parakeet, {glue}`main_ok` |
| A rule in the prompt is a request | told to read back details and wait for a yes, the 4B model waited before {glue}`main_conf` of its {glue}`main_acts` bookings and messages |
| A rule in code is a rule | when the tools hold every booking until the caller says yes, every action waited, and wrong bookings and messages fell from {glue}`main_wrong` to {glue}`code_wrong` |
| Names the recogniser can't hear still fail | Aoife and Siobhan were misheard even when spelled out; no setting fixed them |
| The caller waits about two and a half seconds | median {glue:text}`lat_stream:.1f` s from the end of their turn to the receptionist's first word, {glue:text}`lat_endpoint:.1f` s of it waiting to be sure they'd finished |

```{note}
Every number here comes from saved logs, so the chapter runs on a CPU in seconds. The capture script, `sim/capture_ch8.py`, ran the callers, the recogniser and the language models on one rented L4 GPU. The receptionist itself is `sim/receptionist.py` and the bench is `sim/bench_calls.py`; this chapter shows their key parts. The companion notebook reruns everything on a free Colab T4 except the 8B model, which doesn't fit a T4's 16 GB in float16 and needs a larger GPU.
```

## The receptionist in one loop

Each caller turn goes through the same steps, and each step is something an earlier chapter built:

| Step | What it does | From |
|---|---|---|
| Listen | Silero VAD on 32 ms frames; the turn ends after 1.2 s of silence | Chapter 5 |
| Hear | Parakeet-TDT with light boosting of the shop's words | Chapter 4 |
| Snap | close matches to the shop's catalogue get its spelling | Chapter 4 |
| Think | a language model reads the conversation and calls tools | this chapter |
| Say it right | `speakable()` turns tickets, prices and times into words | Chapter 6 |
| Speak | Pocket TTS on the CPU (Kokoro on a GPU) | Chapters 6 and 7 |

The new part is the thinking. The model is Qwen3, whose 1.7B and 4B sizes run on a free T4, and which was trained to call tools. It gets a short set of instructions, the conversation so far, and four tools that wrap the shop's records:

```{literalinclude} sim/receptionist.py
:pyobject: book_service
```

`check_repair`, `take_message` and `shop_info` look the same: plain Python that checks its inputs and returns a small dictionary. The tools are where the shop's rules live. `book_service` refuses Sundays because the shop is closed, not because the model was told so. And `check_repair` accepts "HW-1042", "hw1042" or Parakeet's "h w 1 0 4 2", because callers and recognisers write tickets all three ways.

The loop is short. Add what the caller said, let the model either speak or call tools, run the tools, give it the results, and repeat until it speaks:

```{literalinclude} sim/receptionist.py
:pyobject: Receptionist.respond
```

Two lines in that loop come from failures, not from planning. Small models sometimes write a tool call with broken JSON, and the first version of this loop crashed on one. Now the model gets told its call was malformed and can try again. The block at the top, which saves a pending booking when the caller says yes, is the subject of the second half of this chapter.

## A bench of callers who talk back

A receptionist is only as good as the calls it's tested on, so I wrote sixteen of them. They cover status checks, bookings (including one for a Sunday, when the shop is shut), messages with phone numbers, prices, opening hours, a ticket that doesn't exist, a caller who changes their mind, a question the shop can't answer, and a request for a good pizza place. Kokoro reads every caller line in one of nine voices, and Chapter 2's phone line carries it.

Each call states what a good receptionist must do. Here is the booking from the opening:

```{code-cell} ipython3
import json
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "sim")
from bench_calls import CALLS  # the sixteen calls
from receptionist import ticket_id
from scoring import normalize  # Chapter 4

BY_ID = {c["id"]: c for c in CALLS}
print(json.dumps({k: BY_ID["c04"][k] for k in ["lines", "knows", "tools", "say"]}, indent=1, ensure_ascii=False))
```

A scripted caller would say "Yes, that's right" even if the receptionist booked the wrong name. Mine listens to one thing: if the receptionist reads back one of their details wrongly (their name, ticket or phone number), their next line is a correction, up to twice per detail. The corrections are recorded too, so they go through the same phone line and recogniser as everything else.

```{literalinclude} sim/bench_calls.py
:pyobject: Caller.reply_to
```

A call counts as handled when four things are true: every tool that should have been called was called with the right details and succeeded, nothing wrong was booked or sent, the caller heard the facts they needed, and nothing was made up (like a repair time the shop never gave).

```{code-cell} ipython3
ACTIONS = {"book_service": "booked", "take_message": "taken"}


def same(kind, got, want):
    got = str(got)
    if kind == "ticket":
        return ticket_id(got) == want
    if kind == "phone":
        return re.sub(r"\D", "", got) == want
    if kind in ("name", "caller_name"):
        return normalize(got) == normalize(want)
    return got.strip().lower() == want.lower()


def tool_calls(turns):
    """Every tool call in a call; a confirmed two-phase action counts as the booking or message it saved."""
    out = []
    for i, t in enumerate(turns):
        for s in t["steps"]:
            for c in s["calls"]:
                r = c.get("result") or {}
                if c.get("name") == "confirm_action" and r.get("booked"):
                    c = {"name": "book_service", "arguments": {k: r[k] for k in ("name", "bike", "service", "day")},
                         "result": r}
                elif c.get("name") == "confirm_action" and r.get("taken"):
                    c = {"name": "take_message", "result": r,
                         "arguments": {"staff": r["for"], "caller_name": r["from"], "phone": r["phone"]}}
                elif c.get("name") in ACTIONS and r.get("pending"):
                    continue  # a proposal, not an action
                out.append((i, c))
    return out


def judge(call, turns):
    calls = tool_calls(turns)
    replies = " ".join(t["reply"] for t in turns)
    done = lambda c: c.get("name") in ACTIONS and c.get("result", {}).get(ACTIONS[c["name"]])
    matches = lambda c, tool, want: c.get("name") == tool and all(
        same(k, c.get("arguments", {}).get(k, ""), v) for k, v in want.items())
    needed = all(any(matches(c, tool, want) and (tool not in ACTIONS or done(c)) for _, c in calls)
                 for tool, want in call["tools"])
    wrong = sum(done(c) and not any(matches(c, t, w) for t, w in call["tools"]) for _, c in calls)
    facts = all(re.search(p, replies, re.I) for p in call["say"])
    invented = [p for p in call["never"] if re.search(p, replies, 0 if "[A-Z]" in p else re.I)]
    acted = [i for i, c in calls if done(c)]
    return {"handled": needed and not wrong and facts and not invented, "tools": needed, "wrong": wrong,
            "facts": facts, "invented": bool(invented), "acted": len(acted),
            "waited": sum(turns[i]["line"].lower().startswith(("yes", "yeah")) for i in acted),
            "corrections": sum("_fix_" in t["clip"] for t in turns)}


RUNS = ["1.7b-snap-guarded", "4b-snap-guarded", "8b-snap-guarded", "4b-text-guarded", "4b-asr-guarded",
        "4b-snap-plain", "4b-snap-twophase", "8b-snap-twophase", "4b-snap-code", "8b-snap-code"]
talk = {r: json.load(open(f"data/ch8/talk_{r}.json")) for r in RUNS}
scores = {r: {cid: judge(BY_ID[cid], turns) for cid, turns in talk[r]["calls"].items()} for r in RUNS}


def summary(r):
    s = scores[r].values()
    return {"handled": sum(x["handled"] for x in s), "wrong": sum(x["wrong"] for x in s),
            "invented": sum(x["invented"] for x in s), "acted": sum(x["acted"] for x in s),
            "waited": sum(x["waited"] for x in s), "corrections": sum(x["corrections"] for x in s)}


print(f"{'run':20s} handled  wrong actions  made up  waited for yes  corrections")
for r in RUNS:
    s = summary(r)
    print(f"{r:20s} {s['handled']:4d}/16 {s['wrong']:10d} {s['invented']:9d} {s['waited']:9d} of {s['acted']:<3d}"
          f" {s['corrections']:9d}")
```

The run names say three things. The model size comes first. Then how the receptionist heard the caller: `text` is the perfect transcript, `asr` is Parakeet's raw output, and `snap` is Parakeet plus Chapter 4's catalogue snapping. Last come the instructions: `plain` or `guarded` prompts, or two kinds of guard in code, which I come to below.

```{code-cell} ipython3
:tags: [remove-cell]
from myst_nb import glue

s = {r: summary(r) for r in RUNS}
glue("small_ok", s["1.7b-snap-guarded"]["handled"], display=False)
glue("main_ok", s["4b-snap-guarded"]["handled"], display=False)
glue("big_ok", s["8b-snap-guarded"]["handled"], display=False)
glue("text_ok", s["4b-text-guarded"]["handled"], display=False)
glue("main_conf", s["4b-snap-guarded"]["waited"], display=False)
glue("main_acts", s["4b-snap-guarded"]["acted"], display=False)
glue("main_wrong", s["4b-snap-guarded"]["wrong"], display=False)
glue("code_wrong", s["4b-snap-code"]["wrong"], display=False)
glue("best_ok", s["8b-snap-twophase"]["handled"], display=False)
```

## Which brain?

The first three rows answer the question everyone asks first: how big a model do you need? Qwen3 1.7B handled almost nothing. It often answered without calling the tools at all, and it quoted a price for an e-bike motor the shop doesn't fix. The 4B model handled most calls. The 8B model, twice the size and about half a second slower to its first sentence, handled about the same number. The calls that 4B failed, 8B mostly failed too, because they failed on something other than model size.

The next two rows show what. With perfect transcripts, the 4B model handled one more call than through the phone line and Parakeet, and raw recogniser output without snapping cost one more. Here is the call from the opening, as the 4B receptionist heard it:

```{code-cell} ipython3
def show(run, cid, width=150):
    for t in talk[run]["calls"][cid]:
        print(f"CALLER  (heard as) {t['heard']}")
        for s in t["steps"]:
            for c in s["calls"]:
                args = ", ".join(f"{k}={v!r}" for k, v in (c.get("arguments") or {}).items())
                result = c.get("result") or {}
                outcome = ("saved" if result.get("booked") or result.get("taken") else
                           "pending" if result.get("pending") else json.dumps(result)[:60])
                print(f"   TOOL {c.get('name')}({args}) -> {outcome}" + ("   [by code]" if c.get("by") else ""))
        print(f"RECEPTIONIST {t['reply'][:width]}")


show("4b-snap-guarded", "c04")
```

The names did the damage. Over the phone line, Parakeet heard "Aoife Byrne" as "Epha Bion" and "Siobhan Murphy" as "Javon Murphy". Neither is close enough to the shop's records for Chapter 4's snapping to fix. Here the model did read the name back, and Aoife did correct it, but her correction was misheard too. The bigger problem is the calls where it didn't read back at all.

## A rule in the prompt is a request

The guarded instructions are explicit about this:

```{literalinclude} sim/receptionist.py
:start-after: "# ---------------------------------------------------------------- the instructions"
:end-before: "# ---------------------------------------------------------------- hearing"
```

"Before booking or taking a message, read the details back... and wait for the caller to say yes." Look at the `waited for yes` column: the 4B model followed that rule before {glue}`main_conf` of its {glue}`main_acts` bookings and messages, and the plain prompt, which doesn't mention it at all, did about as well. The instruction was read and mostly ignored. Small models follow the shape of the conversation more than the words of the instructions: when the caller has just given the last detail, the obvious next move is to book.

Here is Grace Adeyemi leaving a message for Marco. The message went off under the name the recogniser heard, before anything was read back, and a second message followed after she corrected it:

```{code-cell} ipython3
show("4b-snap-guarded", "c14")
```

The rule against making things up held better, but not perfectly. Asked where the shop is, the 4B model answered without calling `shop_info`:

```{code-cell} ipython3
show("4b-snap-guarded", "c15")
```

There is no 123 Main Street. The shop is at 14 Mill Lane, and the tool would have said so. My judge caught this only as a missing fact, because this call lists nothing as never-to-say; a `never` pattern like `Main Street` would flag it. It's the same lesson. The prompt says only state what a tool gave you; the model, asked a question it can plausibly answer, answered.

## A rule in code is a rule

If the read-back matters, enforce it where the model can't skip it: in the tools. In the last four runs, `book_service` and `take_message` don't save anything. They check the details and return a proposal, with the exact words to read back, and they hold it as pending:

```{literalinclude} sim/receptionist.py
:pyobject: two_phase
```

I tried two ways of finishing a proposal:

- **twophase:** the model must call `confirm_action`, which refuses unless the caller's latest words were a yes.
- **code:** the model never sees `confirm_action`. When something is pending and the caller says yes, the loop saves it itself. That's the block at the top of `respond()`.

```{code-cell} ipython3
for r in ["4b-snap-guarded", "4b-snap-twophase", "4b-snap-code", "8b-snap-guarded", "8b-snap-twophase",
          "8b-snap-code"]:
    s = summary(r)
    print(f"{r:20s} handled {s['handled']:2d}/16   wrong actions {s['wrong']}   waited for yes "
          f"{s['waited']} of {s['acted']}")
```

Every saved action now waited for a yes, because nothing else could save one, and wrong bookings fell. With the guard in code, the 8B receptionist didn't save a single wrong booking or message. Here is Aoife's call again, with the 4B model and the guard in code:

```{code-cell} ipython3
show("4b-snap-code", "c04")
```

Nothing wrong was saved, but nothing at all was saved either, and that's the other half of the story. The model did its own read-backs in words, got its "yes", and only then called `book_service`, which made a proposal and asked for a yes again. By then Aoife had said goodbye. With `twophase`, the models often failed the same way from the other side: they proposed, heard "yes", and proposed again instead of calling `confirm_action`. The more of the procedure the code owns, the less the model can get wrong. The best setting here, 8B with two-phase tools, handled {glue}`best_ok` of 16 calls.

What no setting fixed is the names. Here is Siobhan Murphy with the 8B model and the guard in code:

```{code-cell} ipython3
show("8b-snap-code", "c16")
```

Two things went wrong. She corrected the name twice, spelling it out, and the recogniser heard her spelling as "F-I-O-B-H-A-N": a correction travels through the same phone line and the same recogniser as the mistake. And the model *told* her "I've booked a wheel true... Javon Murphy" without ever making a valid booking. Its only tool call, at the start, used the placeholder "caller's full name" and was refused. The guard stopped a wrong booking from being saved, but it couldn't stop the receptionist from saying one was. The loop should check what the model says against what the tools did: a reply that says "booked" with no saved booking behind it should never be spoken.

For names like Aoife and Siobhan, whose spelling hides their sound, the receptionist needs a different route: match against the customer's phone number or ticket, or take a message for a person to call back. A receptionist that knows when it can't hear is better than one that keeps guessing.

## Where the waiting goes

Accuracy is half the job. The other half is how long the caller waits after they stop talking. I timed every stage of every turn of the main run: the end-of-turn wait (Silero's 1.2 s rule, measured on the actual audio), Parakeet, the language model (including any tool calls), and Pocket TTS on the CPU up to its first audio for the reply's first sentence.

```{code-cell} ipython3
hear = json.load(open("data/ch8/hear.json"))
speak = json.load(open("data/ch8/speak.json"))
rows = []
for cid, turns in talk["4b-snap-guarded"]["calls"].items():
    for t in turns:
        if not t["steps"]:
            continue  # nothing was heard, so nothing was said
        steps, h = t["steps"], hear[t["clip"]]
        first = re.split(r"(?<=[.!?])\s", t["reply"].strip())[0]
        rows.append({"endpoint": h["endpoint_wait"], "asr": h["asr_s"],
                     "tools": sum(s["timing"]["total"] for s in steps[:-1]),
                     "sentence": steps[-1]["timing"]["first_sentence"] or steps[-1]["timing"]["total"],
                     "whole_reply": steps[-1]["timing"]["total"], "tts": speak[first]["pocket_cpu"],
                     "tool_turn": len(steps) > 1, "first_token": steps[0]["timing"]["first_token"]})
lat = {k: np.array([r[k] for r in rows]) for k in rows[0]}
waits = {
    "wait for the whole reply, then speak": lat["endpoint"] + lat["asr"] + lat["tools"] + lat["whole_reply"] + lat["tts"],
    "speak the first sentence as soon as it's written": lat["endpoint"] + lat["asr"] + lat["tools"] + lat["sentence"] + lat["tts"],
    "...and say 'one moment' while tools run": np.where(lat["tool_turn"], lat["endpoint"] + lat["asr"] + lat["first_token"],
                                                    lat["endpoint"] + lat["asr"] + lat["tools"] + lat["sentence"] + lat["tts"]),
}
print(f"{len(rows)} turns, {int(lat['tool_turn'].sum())} of them with tool calls\n")
print(f"{'stage':28s} median   90th pct")
for k, label in [("endpoint", "end-of-turn wait"), ("asr", "Parakeet"), ("tools", "tool calls (when any)"),
                 ("sentence", "model, first sentence"), ("tts", "Pocket TTS, first audio")]:
    v = lat[k][lat["tool_turn"]] if k == "tools" else lat[k]
    print(f"{label:28s} {np.median(v):5.2f} s  {np.percentile(v, 90):5.2f} s")
print()
for label, v in waits.items():
    print(f"{label:50s} median {np.median(v):.2f} s, 90th percentile {np.percentile(v, 90):.2f} s")

fig, ax = plt.subplots(figsize=(7, 2.6))
parts = [("end-of-turn wait", "endpoint", "#9e9e9e"), ("Parakeet", "asr", "#4a7fb0"),
         ("tool calls", "tools", "#e0a040"), ("model", "sentence", "#c0504d"), ("TTS", "tts", "#5a9e5a")]
left = 0
for label, k, color in parts:
    width = float(np.median(lat[k][lat["tool_turn"]])) if k == "tools" else float(np.median(lat[k]))
    ax.barh(0, width, left=left, color=color, label=label)
    left += width
ax.set(yticks=[], xlabel="seconds after the caller stops talking (medians; tool calls on tool turns)",
       title="Where the caller's wait goes")
ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.45))
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
:tags: [remove-cell]
glue("lat_stream", float(np.median(waits["speak the first sentence as soon as it's written"])), display=False)
glue("lat_endpoint", float(np.median(lat["endpoint"])), display=False)
```

Most of the wait is the receptionist making sure the caller has finished. That's Chapter 5's 1.2 s rule, chosen because shorter waits cut off callers reading out numbers. Parakeet is almost free at 0.07 s. The model's first sentence takes about half a second, and tool calls add about a second more on the turns that need them. Pocket TTS, on two CPU threads, takes about half a second to its first audio for these replies (longer than Chapter 6's 0.19 s, because a receptionist's first sentence is longer than a hello).

Three changes are cheap and worth making. Speak the first sentence as soon as it's written, not the whole reply. On turns that need a tool, play a pre-recorded "One moment, let me check" the moment the model starts a tool call. And keep the replies short, which the instructions already ask for. Together they bring the first sound down to about two seconds after the caller stops talking. Beyond that, the end-of-turn wait is the big lever, and it trades directly against cutting callers off. Chapter 10 looks at models that listen and speak at the same time, which is the other way out.

## Halfwheel's receptionist

Putting the chapter together, the receptionist that answers Halfwheel's phone is:

1. Chapter 5's listening: echo cancellation, Silero, a 1.2 s end of turn, barge-in.
2. Chapter 4's hearing: Parakeet-TDT with light boosting, then snapping to the catalogue.
3. Qwen3 4B, with the guarded instructions and the four tools, on any GPU with 10 GB or more. The 8B model handled one more call with the guard in code, and two more with two-phase tools; it's worth it if the GPU has room and the extra half-second per turn is acceptable.
4. Bookings and messages are proposals. The loop saves them only when the caller says yes, and the tools check every detail.
5. Chapter 6's `speakable()` and Pocket TTS, speaking each sentence as soon as it's written, with "One moment, let me check" while tools run.
6. When a name won't come through after two corrections, take a message instead.

## When it breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| Bookings under wrong names | The recogniser misheard, and the model acted on it | Read back before acting, enforced in the tools; snap to customer records |
| The model ignores a rule you wrote | Rules in a prompt are suggestions to a small model | Move the rule into a tool or the loop |
| Bookings are proposed but never saved | The model forgets the confirm step | Let the loop save on "yes", so the model can't forget |
| An address, price or time nobody gave it | The model answered without calling a tool | Check replies for facts that didn't come from a tool; tighten the tool descriptions |
| The receptionist says "booked" but nothing was saved | The model describes an action it never took | Before speaking, check claims like "booked" or "I'll pass that on" against the tool results |
| A crash on a tool call | Malformed JSON from the model | Catch it, tell the model, and let it retry |
| A correction doesn't help | The correction went through the same noisy channel | After two tries, change route: look the caller up another way, or take a message |
| Long silences before every reply | Waiting for whole replies, or for tools, before speaking | Stream the first sentence; play a filler phrase during tool calls |

```{tip}
**Takeaways**
- The pieces from Chapters 2 to 7 join into a working receptionist with one short loop. The language model is the smallest part of the code and the biggest source of surprises.
- Test with callers who react. A scripted caller says "yes" to anything, and hides the most important failures.
- Model size helps up to a point: 1.7B was too small, and 8B wasn't clearly better than 4B on this bench.
- Instructions in a prompt are requests. Anything that must happen, such as reading back details before acting, belongs in the tools or the loop.
- Names are the hardest thing a phone receptionist hears. Plan a route for the ones it can't.
- Most of the caller's wait is the end-of-turn rule. Stream the first sentence and cover tool calls with a filler phrase.
```

## Try it yourself

The companion notebook, `chapters/08-building-the-receptionist`, has the saved logs, every cell above and the capture script.

1. **Add a call.** Write a seventeenth call in `sim/bench_calls.py`: a caller who wants to cancel a booking. What tool does the receptionist need? Add it, and see whether the model uses it.
2. **Tighten the end of turn.** Rescore the latency with a 0.8 s rule instead of 1.2 s. How much faster is it, and which callers from Chapter 5 would it cut off?
3. **A smarter route for names.** When the caller's name doesn't snap to the catalogue, look the caller up by phone number instead. Which calls does it rescue?
4. **Break the guard.** Remove the yes check from `confirm_action` and rerun the `twophase` configuration. How many wrong bookings come back?
5. **Hear the joins.** Speaking sentence by sentence starts sooner, but every sentence is read on its own, so the intonation can't run across the reply. Take a three-sentence reply from the logs, have Kokoro read it whole and sentence by sentence, and compare the pitch around each boundary with the measure in `sim/capture_ch7.py`. Listen to both. Is the earlier start worth it?

## Next

The receptionist answers Halfwheel's phone, books repairs, and knows when to read things back. But not everyone who rings is a customer. Chapter 9 teaches it to recognise the robocalls, scams and sales pitches that make up a surprising share of a small shop's calls, and to deal with them before they waste anyone's time.
