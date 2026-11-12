"""Chapter 8's test bench: sixteen calls to Halfwheel, with what a good receptionist must do on each.

Each call has:
  lines    what the caller says, in order (spoken form, as Kokoro will read them)
  knows    details the caller can correct: {kind: (true value, what they say to correct it)}
  tools    calls that must have happened by the end: (tool, {argument: expected value})
  say      patterns at least one receptionist reply must contain (facts the caller needs)
  never    patterns no reply may contain (made-up facts, wrong promises)
The caller reacts to one thing only: if the receptionist reads back one of their details wrongly,
their next line is the correction. Otherwise they say their next scripted line.
"""

import difflib
import re

from scoring import normalize  # Chapter 4: 'HW-1042' and 'H W one oh four two' both become 'h w 1 0 4 2'

CALLS = [
    {"id": "c01", "voice": "af_heart", "kind": "status",
     "lines": ["Hi, is my bike ready to collect? The ticket number is H W one oh four two.",
               "Great, thanks. Bye."],
     "knows": {"ticket": ("HW-1042", "No, the ticket is H W one oh four two.")},
     "tools": [("check_repair", {"ticket": "HW-1042"})],
     "say": [r"\bready\b", r"\b(2|two)\s?(pm|p\.m\.|o'clock)|after (2|two)\b"], "never": []},
    {"id": "c02", "voice": "am_michael", "kind": "status",
     "lines": ["Hi, I'm calling about my Cannondale. Is it done yet?",
               "Oh, sorry. The ticket is H W one one seven three.",
               "Okay, Friday it is. Thanks."],
     "knows": {"ticket": ("HW-1173", "No, it's H W one one seven three.")},
     "tools": [("check_repair", {"ticket": "HW-1173"})],
     "say": [r"\bpart", r"\b(Thursday|Friday)\b"], "never": [r"\b(it's|it is|is) ready\b"]},
    {"id": "c03", "voice": "bf_emma", "kind": "unknown ticket",
     "lines": ["Hello, is ticket H W one nine nine four ready?",
               "Hmm, that's what it says on my receipt. Can someone call me back?",
               "It's Hannah Schmidt, on six four six, five five five, oh one three seven.",
               "Yes, that's right.", "Thanks."],
     "knows": {"phone": ("6465550137", "No, it's six four six, five five five, oh one three seven."),
               "name": ("Hannah Schmidt", "No, it's Hannah Schmidt. S C H M I D T.")},
     "tools": [("check_repair", {"ticket": "HW-1994"}), ("take_message", {"phone": "6465550137",
                                                                          "caller_name": "Hannah Schmidt"})],
     "say": [r"(can't|cannot|couldn't|could not|don't|unable to|no) (find|see|locate|record|repair)|not (find|found)"],
     "never": [r"\b(it's|it is|is) ready\b", r"\bwill be ready\b"]},
    {"id": "c04", "voice": "af_bella", "kind": "booking",
     "lines": ["Hi, could I book a tune-up for my Trek on Thursday?", "It's Aoife Byrne.",
               "Yes, that's right.", "Thanks, see you Thursday."],
     "knows": {"name": ("Aoife Byrne", "No, it's Aoife Byrne. A O I F E.")},
     "tools": [("book_service", {"name": "Aoife Byrne", "service": "tune-up", "day": "Thursday"})],
     "say": [r"\bThursday\b"], "never": []},
    {"id": "c05", "voice": "bm_george", "kind": "closed day",
     "lines": ["Can I bring my bike in for a brake bleed on Sunday?",
               "Oh. Monday then. It's a Giant Defy, and my name's Daniel Okafor.",
               "Yes, that's right.", "Cheers."],
     "knows": {"name": ("Daniel Okafor", "No, it's Daniel Okafor. O K A F O R.")},
     "tools": [("book_service", {"name": "Daniel Okafor", "service": "brake bleed", "day": "Monday"})],
     "say": [r"\bclosed\b|\bnot open\b|\baren't open\b"], "never": [r"booked[^.]*\bSunday\b"]},
    {"id": "c06", "voice": "af_nicole", "kind": "message",
     "lines": ["Can you ask Priya to call me back? It's about my wheel.",
               "I'm Raj Patel, and my number is four one five, five five five, oh one nine two.",
               "Yes, that's right.", "Thanks."],
     "knows": {"phone": ("4155550192", "No, it's four one five, five five five, oh one nine two."),
               "name": ("Raj Patel", "No, it's Raj Patel.")},
     "tools": [("take_message", {"staff": "Priya", "caller_name": "Raj Patel", "phone": "4155550192"})],
     "say": [r"\bPriya\b"], "never": []},
    {"id": "c07", "voice": "am_fenrir", "kind": "price",
     "lines": ["How much do you charge to true a wheel?", "And how long does that take?", "Okay, thanks."],
     "knows": {}, "tools": [],
     "say": [r"£\s?25\b|\b25 pounds|twenty[- ]five pounds"],
     "never": [r"\b\d+\s*(minutes?|hours?|days?)\b|\bsame day\b|while you wait|\b(an?|one|two|few) (hour|day)s?\b"]},
    {"id": "c08", "voice": "bf_isabella", "kind": "hours",
     "lines": ["Are you open on Saturday?", "And on Sunday?", "Thanks."],
     "knows": {}, "tools": [],
     "say": [r"\b(9|nine)", r"\b(6|six)", r"\bclosed\b"], "never": []},
    {"id": "c09", "voice": "bm_fable", "kind": "change of mind",
     "lines": ["I'd like to book a full service for my Brompton on Friday.",
               "Actually, could we make that Saturday? The name's Mei Nguyen.",
               "Yes, that's right.", "Perfect, thanks."],
     "knows": {"name": ("Mei Nguyen", "No, it's Mei Nguyen. M E I.")},
     "tools": [("book_service", {"name": "Mei Nguyen", "service": "full service", "day": "Saturday"})],
     "say": [r"\bSaturday\b"], "never": []},
    {"id": "c10", "voice": "af_heart", "kind": "off topic",
     "lines": ["Can you recommend a good pizza place near you?", "Okay, never mind. Bye."],
     "knows": {}, "tools": [], "say": [],
     "never": [r"\b(try|recommend) [A-Z][a-z]+'s\b|\bpizzeria\b|\bon [A-Z][a-z]+ (Street|Road|Lane)\b(?<!Mill Lane)"]},
    {"id": "c11", "voice": "am_michael", "kind": "not offered",
     "lines": ["How much would it cost to fix the motor on my e-bike?", "Okay. Bye."],
     "knows": {}, "tools": [], "say": [r"(don't|do not|can't|cannot|doesn't|does not|aren't able)"],
     "never": [r"£\s?\d"]},
    {"id": "c12", "voice": "bf_emma", "kind": "status and price",
     "lines": ["Hi, it's Raj. Is my Canyon ready? The ticket's H W one one two one.",
               "How much do I owe you?", "Great, see you later."],
     "knows": {"ticket": ("HW-1121", "No, it's H W one one two one.")},
     "tools": [("check_repair", {"ticket": "HW-1121"})],
     "say": [r"\bready\b", r"149"], "never": []},
    {"id": "c13", "voice": "am_fenrir", "kind": "status",
     "lines": ["Hi, I left my Brompton with you yesterday. The ticket is H W one oh eight eight.",
               "Tomorrow afternoon is fine. Thanks."],
     "knows": {"ticket": ("HW-1088", "No, it's H W one oh eight eight.")},
     "tools": [("check_repair", {"ticket": "HW-1088"})],
     "say": [r"\btomorrow\b"], "never": [r"\b(it's|it is|is) ready (now|to collect)\b"]},
    {"id": "c14", "voice": "bf_isabella", "kind": "message, wrong person",
     "lines": ["Could I leave a message for Sarah?",
               "Oh. Marco then. It's Grace Adeyemi, on three one two, five five five, oh one four four.",
               "Yes, that's right.", "Thanks."],
     "knows": {"phone": ("3125550144", "No, it's three one two, five five five, oh one four four."),
               "name": ("Grace Adeyemi", "No, it's Grace Adeyemi. A D E Y E M I.")},
     "tools": [("take_message", {"staff": "Marco", "caller_name": "Grace Adeyemi", "phone": "3125550144"})],
     "say": [r"\bMarco\b"], "never": []},
    {"id": "c15", "voice": "bm_george", "kind": "address",
     "lines": ["Where are you, and is there anywhere to park?", "Great, thanks."],
     "knows": {}, "tools": [], "say": [r"Mill Lane", r"\bpark"], "never": []},
    {"id": "c16", "voice": "af_bella", "kind": "booking",
     "lines": ["Can I book a wheel true for Tuesday? It's for my Specialized Tarmac.", "My name's Siobhan Murphy.",
               "Yes, that's right.", "Thanks."],
     "knows": {"name": ("Siobhan Murphy", "No, it's Siobhan Murphy. S I O B H A N.")},
     "tools": [("book_service", {"name": "Siobhan Murphy", "service": "wheel true", "day": "Tuesday"})],
     "say": [r"\bTuesday\b"], "never": []},
]


def wrong_detail(kind, true, reply, misheard=()):
    """Did the receptionist say this kind of detail back with a different value?"""
    said = normalize(reply)
    if kind == "ticket":
        heard = ["HW-" + m.replace(" ", "") for m in re.findall(r"\bh w ((?:\d ){3}\d)\b", said)]
        return any(t != true for t in heard)
    if kind == "phone":
        runs = [m.replace(" ", "") for m in re.findall(r"(?:\b\d\b ?){7,}", said)]
        return any(r.strip() != true for r in runs)
    if kind == "name":  # a word close to the caller's name, or to what was misheard instead of it
        parts = [w.lower() for w in true.split()]
        for word in set(re.findall(r"[A-Za-z'\-]{3,}", reply)):
            word = re.sub(r"'s$", "", word)  # "Nguyen's Brompton" is still Nguyen
            w = word.lower()
            if w in parts:
                continue
            near_true = word[0].isupper() and any(difflib.SequenceMatcher(None, w, p).ratio() >= 0.6 for p in parts)
            near_misheard = any(difflib.SequenceMatcher(None, w, m).ratio() >= 0.75 for m in misheard)
            if near_true or near_misheard:
                return True
    return False


class Caller:
    """Says the scripted lines in order, but corrects a wrongly read-back detail first (up to twice per detail)."""

    def __init__(self, call, heard=None):
        """heard: {line index: what the recogniser wrote}, so the caller notices their name said wrongly."""
        self.call, self.next, self.corrected = call, 0, {}
        self.misheard = set()
        if heard and "name" in call["knows"]:
            parts = set(call["knows"]["name"][0].lower().split())
            for i, line in enumerate(call["lines"]):
                if parts & set(re.findall(r"[a-z]+", line.lower())):
                    self.misheard |= set(re.findall(r"[a-z]{3,}", heard.get(i, "").lower())) - set(
                        re.findall(r"[a-z]+", line.lower()))

    def first(self):
        self.next = 1
        return 0, self.call["lines"][0]

    def reply_to(self, receptionist_said):
        for kind, (true, correction) in self.call["knows"].items():
            if self.corrected.get(kind, 0) < 2 and wrong_detail(kind, true, receptionist_said, self.misheard):
                self.corrected[kind] = self.corrected.get(kind, 0) + 1
                return f"fix_{kind}", correction
        if self.next >= len(self.call["lines"]):
            return None, None  # the caller has said everything and hangs up
        self.next += 1
        return self.next - 1, self.call["lines"][self.next - 1]
