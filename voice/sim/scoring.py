"""Transcript scoring used before Chapter 4 explains it.

These are the same functions Chapter 4 builds and explains line by line:
word error rate as an edit distance over words, a normalizer that removes
formatting differences, and entity hits. Chapters 2 and 3 import them from here.
"""

import re

import numpy as np

SMALL = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
SMALL["oh"] = 0
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
STOP = set("a i is it on to my me of in at be do so up or an as by if no we he us am hi ok".split())


def edit_ops(ref, hyp):
    """Minimum (substitutions, deletions, insertions) turning ref into hyp."""
    R, H = len(ref), len(hyp)
    d = np.zeros((R + 1, H + 1), dtype=int)
    d[:, 0], d[0, :] = np.arange(R + 1), np.arange(H + 1)
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            d[i, j] = min(d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1, j] + 1, d[i, j - 1] + 1)
    i, j, ops = R, H, {"sub": 0, "del": 0, "ins": 0}
    while i or j:
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


def normalize(text):
    """One canonical form for scoring: 'HW-1042' and 'h w one oh four two' both become 'h w 1 0 4 2'."""
    t = text.lower().replace("-", " ").replace("'", "")
    toks = re.sub(r"[^a-z0-9 ]", " ", t).split()
    out, i = [], 0
    while i < len(toks):
        tok = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if tok in TENS and nxt in SMALL and 0 < SMALL[nxt] < 10:
            out += list(str(TENS[tok] + SMALL[nxt]))
            i += 2
            continue
        if tok.isdigit() or re.fullmatch(r"[a-z]+\d+|\d+[a-z]+", tok):
            out += list(tok)
        elif tok in SMALL:
            out += list(str(SMALL[tok]))
        elif tok == "o" and ((out and out[-1].isdigit()) or nxt in SMALL or nxt.isdigit()):
            out.append("0")
        elif tok in TENS:
            out += list(str(TENS[tok]))
        elif len(tok) <= 2 and tok not in STOP:
            out += list(tok)
        else:
            out.append(tok)
        i += 1
    return " ".join(out)


def word_errors(ref, hyp):
    """(errors, reference words) after normalizing both sides."""
    r, h = normalize(ref).split(), normalize(hyp).split()
    return sum(edit_ops(r, h).values()), len(r)


def entity_hits(call, hyp):
    """[(entity type, heard?)] for every entity the call's manifest row lists."""
    h = f" {normalize(hyp)} "
    return [(e["type"], f" {normalize(e['written'])} " in h) for e in call.get("entities", [])]
