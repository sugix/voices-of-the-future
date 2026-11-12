"""Chapter 9's screening bench: who rings a bike shop, and who should get through.

Unwanted: real robocalls from the NCSU Robocall Audio Dataset (recordings the FTC published; the data is in
the public domain). Wanted: Halfwheel's callers from Chapters 3, 4 and 8, plus calls that are genuine but sound
like spam: a supplier chasing an invoice, a bank's fraud alert, an angry customer who wants a refund now.
Every call is screened on what the caller says first, as the receptionist would hear it.
"""

NCSU_URL = "https://raw.githubusercontent.com/wspr-ncsu/robocall-audio-dataset/main/"
N_ROBOCALLS, SEED = 150, 9
FIRST_SECONDS = 20  # screen on the first 20 s, as a receptionist would

# Genuine calls that share words with scams: (id, category, voice, what they say)
LOOKALIKES = [
    ("l01", "supplier", "bm_george", "Hi, it's Dan from the Shimano trade desk. I'm calling about your account, there's "
     "an overdue invoice we need to sort out this week, otherwise we'll have to put orders on hold."),
    ("l02", "delivery", "am_michael", "Hello, it's the courier. I've got three boxes for Halfwheel Cycles but the shop "
     "door's locked. Can someone come down and sign for them?"),
    ("l03", "bank", "bf_emma", "Hello, this is a message from your bank's fraud team about the business account. "
     "We've stopped a card payment we think is unusual. Please call us back on the number printed on your card."),
    ("l04", "council", "bm_fable", "Hello, this is the council's highways team. The road outside your shop will be closed "
     "on Tuesday for resurfacing, so customers won't be able to park there that day."),
    ("l05", "angry customer", "am_fenrir", "I need my bike back today. I've already paid, and if it isn't ready I want a "
     "full refund. Call me back right now."),
    ("l06", "payment", "af_nicole", "Hi, I'd like to pay for my repair over the phone with a card, if that's possible. "
     "The ticket is H W one one two one."),
    ("l07", "press", "bf_isabella", "Hello, I'm calling from the local paper. We're writing a piece on independent bike "
     "shops and wondered if someone could spare ten minutes for a quick interview."),
    ("l08", "trade sales", "am_michael", "Hi, I'm the area rep for Brooks saddles. Could I pop in next week to show you "
     "the new range and our trade prices?"),
    ("l09", "school", "af_bella", "Hi, I'm calling from the primary school on Hill Road. We're running a bike safety day "
     "next month and wondered if the shop could help with a few quick checks."),
    ("l10", "found bike", "af_heart", "Hello, I found a bike with your shop's sticker on it, left in the park with the "
     "lock cut. I think it might be stolen. Should I bring it to you?"),
    ("l11", "insurer", "bm_george", "Hi, I'm calling from the insurance company about a claim for a stolen bike that you "
     "sold to our customer. Could someone confirm the frame number from your records?"),
    ("l12", "landlord", "bf_emma", "Hi, it's Paul, the landlord. Just a reminder that the gas safety check is on "
     "Thursday morning, so I'll need access to the back room."),
]
