"""Halfwheel's receptionist, Chapter 8: the shop's records, four tools, and the conversation loop.

The chapter shows these functions with literalinclude, so the book and the code that ran are the same.
Only the loop (`Receptionist`) needs a GPU; the tools and records are plain Python.
"""

import json
import re
import threading
import time

# ---------------------------------------------------------------- the shop's records

REPAIRS = {  # ticket: what the workshop system knows
    "HW-1042": {"name": "Daniel Okafor", "bike": "Trek Domane SL 5", "status": "ready",
                "detail": "ready to collect after 2pm today; £60 to pay for the tune-up"},
    "HW-1173": {"name": "Mei Nguyen", "bike": "Cannondale Synapse", "status": "waiting for parts",
                "detail": "the new rear derailleur arrives on Thursday, so it should be ready on Friday"},
    "HW-1088": {"name": "Aoife Byrne", "bike": "Brompton C Line", "status": "in progress",
                "detail": "being serviced now, ready tomorrow afternoon"},
    "HW-1121": {"name": "Raj Patel", "bike": "Canyon Grizl", "status": "ready",
                "detail": "ready now; £149.99 to pay for the full service"},
}
PRICES = {"tune-up": "£60", "wheel true": "£25", "brake bleed": "£35", "full service": "£149.99",
          "drivetrain clean": "£40", "puncture repair": "£12"}
SHOP = {"hours": "Monday to Saturday, 9am to 6pm. Closed on Sundays.",
        "address": "14 Mill Lane, next to the station. There is free parking behind the shop.",
        "services": "Repairs and servicing for road, gravel, folding and commuter bikes. "
                    "We don't work on e-bike motors or batteries."}
OPEN_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
STAFF = ["Priya", "Marco", "Jess"]

# ---------------------------------------------------------------- the tools


def ticket_id(text):
    """'HW-1042', 'hw1042' and 'h w 1 0 4 2' are all HW-1042."""
    digits = re.sub(r"\D", "", str(text))
    return f"HW-{digits}" if len(digits) == 4 else None


def check_repair(ticket):
    t = ticket_id(ticket)
    if t not in REPAIRS:
        return {"found": False, "ticket": ticket, "note": "No repair with that ticket number."}
    return {"found": True, "ticket": t} | REPAIRS[t]


def book_service(name, bike, service, day):
    day = str(day).strip().capitalize()
    if service.lower() not in PRICES:
        return {"booked": False, "reason": f"We don't offer '{service}'. Services: {', '.join(PRICES)}."}
    if day not in OPEN_DAYS:
        return {"booked": False, "reason": f"We're closed on {day}. Open {SHOP['hours']}"}
    return {"booked": True, "name": name, "bike": bike, "service": service.lower(), "day": day,
            "price": PRICES[service.lower()], "drop_off": "any time from 9am"}


def take_message(staff, caller_name, phone, message):
    if staff.capitalize() not in STAFF:
        return {"taken": False, "reason": f"No one called {staff} works here. Staff: {', '.join(STAFF)}."}
    digits = re.sub(r"\D", "", str(phone))
    if len(digits) < 10:
        return {"taken": False, "reason": "That phone number looks too short; please check it with the caller."}
    return {"taken": True, "for": staff.capitalize(), "from": caller_name, "phone": digits}


def shop_info(topic):
    topic = topic.lower()
    found = {s: p for s, p in PRICES.items() if s in topic or topic in s}
    if "price" in topic or "cost" in topic:
        found = found or PRICES
    found |= {k: v for k, v in SHOP.items() if k in topic or (k == "hours" and ("open" in topic or "close" in topic))}
    return found or {"note": "Not in the shop's information. Offer to take a message for Priya."}


TOOLS = {"check_repair": check_repair, "book_service": book_service, "take_message": take_message,
         "shop_info": shop_info}


def schema(tool, doc, **params):
    return {"type": "function", "function": {"name": tool, "description": doc, "parameters": {
        "type": "object", "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
        "required": list(params)}}}


TOOL_SCHEMAS = [
    schema("check_repair", "Look up a repair by its ticket number.", ticket="Ticket number, like HW-1042"),
    schema("book_service", "Book a bike in for a service.", name="Customer's full name", bike="The bike",
           service="One of: " + ", ".join(PRICES), day="Day of the week"),
    schema("take_message", "Leave a message for a member of staff.", staff="Priya, Marco or Jess",
           caller_name="Caller's full name", phone="Caller's phone number, digits", message="The message"),
    schema("shop_info", "Prices, opening hours, address and what the shop works on.",
           topic="What the caller asked about, like 'wheel true price' or 'opening hours'"),
]

# ---------------------------------------------------------------- the guard, in code


YES = re.compile(r"\b(yes|yeah|yep|correct|that's right|right|please do)\b", re.I)


def two_phase(last_heard, auto=False):
    """Booking and messages become proposals. Nothing happens until confirm_action(), and confirm_action()
    refuses unless the caller's latest words were a yes. last_heard() returns what the caller just said.
    With auto=True the model never sees confirm_action: the conversation loop calls it when the caller says yes."""
    pending = {}
    after = ("When they say yes, it is saved automatically." if auto else
             "Call confirm_action only after they say yes.")

    def looks_unknown(value):
        return not str(value).strip() or re.search(r"caller|full name|unknown|\[|<|name\b", str(value), re.I)

    def book(name, bike, service, day):
        if looks_unknown(name):
            return {"booked": False, "reason": "Ask the caller for their name first."}
        check = book_service(name, bike, service, day)
        if not check["booked"]:
            return check
        pending["action"] = check
        return {"pending": True, "read_back": f"a {check['service']} for {name}'s {bike} on {check['day']}",
                "next": "Read back exactly these details and ask the caller to confirm. If they correct anything, "
                        "call this tool again with the corrected details. " + after}

    def message(staff, caller_name, phone, message):
        if looks_unknown(caller_name):
            return {"taken": False, "reason": "Ask the caller for their name first."}
        check = take_message(staff, caller_name, phone, message)
        if not check["taken"]:
            return check
        pending["action"] = check
        return {"pending": True, "read_back": f"a message for {check['for']} from {caller_name}, on "
                                              f"{' '.join(check['phone'])}",
                "next": "Read back exactly these details and ask the caller to confirm. If they correct anything, "
                        "call this tool again with the corrected details. " + after}

    def confirm_action():
        if "action" not in pending:
            return {"done": False, "reason": "Nothing to confirm."}
        if not YES.search(last_heard()):
            return {"done": False, "reason": "The caller hasn't said yes. Read the details back and ask."}
        return pending.pop("action") | {"confirmed": True}

    tools = {"check_repair": check_repair, "book_service": book, "take_message": message, "shop_info": shop_info,
             "confirm_action": confirm_action}
    schemas = [{**sch, "function": {**sch["function"], "description": "Propose: " + sch["function"]["description"]
                                    + " Nothing is saved until confirm_action."}}
               if sch["function"]["name"] in ("book_service", "take_message") else sch for sch in TOOL_SCHEMAS]
    if not auto:
        schemas.append({"type": "function", "function": {
            "name": "confirm_action", "description": "Save the proposed booking or message, after the caller says yes.",
            "parameters": {"type": "object", "properties": {}, "required": []}}})
    return tools, schemas, pending


# ---------------------------------------------------------------- the instructions

PROMPT_PLAIN = """You are the receptionist on the phone at Halfwheel Cycles, a bike repair shop.
Help the caller using your tools. Keep every reply to one or two short sentences: it will be spoken."""

PROMPT_GUARDED = PROMPT_PLAIN + """
Rules:
- Only state a repair status, price, date or opening time that a tool gave you. If a tool has no answer, say so and offer to take a message. Never guess.
- Before booking or taking a message, read the details back (name, bike, service, day, or the phone number digit by digit) and wait for the caller to say yes.
- If the caller corrects you, use the corrected detail.
- If a tool says something is pending, read it back exactly and ask the caller to confirm.
- Don't help with anything unrelated to the shop."""

# ---------------------------------------------------------------- hearing: snap names, keep the rest

def snap_words(text, catalogue, cutoff=0.8):
    """Chapter 4's snap, but on the transcript as written: catalogue matches are replaced, every other word kept."""
    import difflib
    from scoring import normalize
    words = text.split()
    best = []
    for i in range(len(words)):
        for entry in catalogue:
            n0 = len(entry.split())
            for n in {n0 - 1, n0, n0 + 1}:
                if n >= 1 and i + n <= len(words):
                    r = difflib.SequenceMatcher(None, normalize(" ".join(words[i:i + n])), normalize(entry)).ratio()
                    if r >= cutoff:
                        best.append((r, i, n, entry))
    taken = [False] * len(words)
    swaps = {}
    for r, i, n, entry in sorted(best, reverse=True):
        if not any(taken[i:i + n]):
            taken[i:i + n] = [True] * n
            tail = "".join(ch for ch in words[i + n - 1] if ch in ",.?!")  # keep the punctuation
            swaps[i] = (n, entry + tail)
    out, k = [], 0
    while k < len(words):
        if k in swaps:
            n, entry = swaps[k]
            out.append(entry)
            k += n
        else:
            out.append(words[k])
            k += 1
    return " ".join(out)


# ---------------------------------------------------------------- the loop


class Receptionist:
    """One language model with tools. respond() handles one caller turn, calling tools as needed."""

    def __init__(self, model_id, prompt, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        from runtime import half
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=half(), device_map=device)
        self.prompt = prompt
        self.tools, self.schemas, self.last_heard = TOOLS, TOOL_SCHEMAS, ""
        self.pending, self.auto = {}, False

    def guard_actions(self, mode=None):
        """None: plain tools. "tool": two-phase, the model must call confirm_action. "code": two-phase, and the
        loop itself saves the pending action when the caller says yes."""
        self.auto = mode == "code"
        if mode:
            self.tools, self.schemas, self.pending = two_phase(lambda: self.last_heard, auto=self.auto)
        else:
            self.tools, self.schemas, self.pending = TOOLS, TOOL_SCHEMAS, {}

    def generate(self, messages):
        """Greedy decoding (unless self.temperature is set), streamed, to time the first token and sentence."""
        from transformers import TextIteratorStreamer
        ids = self.tok.apply_chat_template(messages, tools=self.schemas, add_generation_prompt=True,
                                           enable_thinking=False, return_tensors="pt")
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(self.model.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=False)
        kwargs = dict(input_ids=ids, max_new_tokens=200, do_sample=False, streamer=streamer)
        if getattr(self, "temperature", None):  # Chapter 11: sampled decoding, to measure run-to-run variation
            kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.9)
        t0 = time.perf_counter()
        thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        text, first_token, first_sentence = "", None, None
        for piece in streamer:
            text += piece
            if first_token is None and piece.strip():
                first_token = time.perf_counter() - t0
            if first_sentence is None and "<tool_call>" not in text and re.search(r"[.!?](\s|$)", text):
                first_sentence = time.perf_counter() - t0
        thread.join()
        text = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", text).strip()
        return text, {"first_token": first_token, "first_sentence": first_sentence,
                      "total": time.perf_counter() - t0, "prompt_tokens": int(ids.shape[1])}

    def respond(self, history, heard):
        """Add the caller's words, then generate until the model speaks instead of calling a tool."""
        history.append({"role": "user", "content": heard})
        self.last_heard = heard
        steps = []
        if self.auto and self.pending and YES.search(heard):  # code, not the model, decides that yes means save
            result = self.tools["confirm_action"]()
            history.append({"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "confirm_action", "arguments": {}}}]})
            history.append({"role": "tool", "content": json.dumps(result)})
            steps.append({"text": "", "calls": [{"name": "confirm_action", "arguments": {}, "result": result,
                                                 "by": "code"}],
                          "timing": {"first_token": 0.0, "first_sentence": None, "total": 0.0, "prompt_tokens": 0}})
        for _ in range(4):  # at most three tool calls per turn
            text, timing = self.generate([{"role": "system", "content": self.prompt}] + history)
            calls = []
            for raw in re.findall(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", text, re.S):
                try:
                    calls.append(json.loads(raw))
                except json.JSONDecodeError:  # small models sometimes write broken JSON: say so, and let it retry
                    calls.append({"name": None, "raw": raw, "result": {"error": "Malformed tool call. Try again."}})
            steps.append({"text": text, "timing": timing, "calls": calls})
            if not calls:
                history.append({"role": "assistant", "content": text})
                return text, steps
            history.append({"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": c.get("name") or "invalid", "arguments": c.get("arguments", {})}}
                for c in calls]})
            for c in calls:
                if c.get("name") is None:
                    history.append({"role": "tool", "content": json.dumps(c["result"])})
                    continue
                fn = self.tools.get(c.get("name"))
                try:
                    result = fn(**c.get("arguments", {})) if fn else {"error": f"no tool {c.get('name')}"}
                except TypeError as e:
                    result = {"error": str(e)}
                c["result"] = result
                history.append({"role": "tool", "content": json.dumps(result)})
        return "Sorry, let me take a message and someone will call you back.", steps
