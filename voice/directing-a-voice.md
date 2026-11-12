---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
date: "2026-12-21"  # quoted: myst-nb stores front matter as JSON
description: Pace, pronunciation, emotion and cloned voices, measured across Kokoro, Kyutai TTS and Fish Audio's S1-mini and S2-Pro. Which controls actually move the voice, and which only look like they should.
---

# Directing a voice

Halfwheel's receptionist now says the right words, and says them the same way every time. "Good news! Your Brompton is ready" and "I'm really sorry, we've lost the front wheel from your Canyon" come out in the same bright, even voice, as if losing a customer's wheel were good news too. A person at the counter would drop their voice for the second one, slow down, and sound like they meant it.

Chapter 6 asked whether a model says the right words. This chapter asks whether you can tell it *how* to say them. I tried every control the open models offer, on five moments from a receptionist's day:

- good news
- an apology
- calming an annoyed caller
- reading back a ticket and a time
- a quick "one moment"

Then I measured what changed: pace, pitch, pauses, whose voice it was, and whether the names came out right.

Here is where I ended up:

| Finding | Number |
|---|---|
| Speed controls work exactly | Kokoro's `speed` took the same read-back from 2.5 to 4.1 words a second, in step with the setting |
| Written stage directions mostly don't | Fish S2-Pro's `[slowly and clearly]` came out *faster* than no tag; S1-mini's `(in a hurry tone)` came out slower |
| The reference clip sets the delivery | cloned from a sad recording, Fish S2-Pro's apology sat at 205 Hz with a 3.1-semitone range; from a neutral one, 238 Hz and 7.7 semitones. The tag moved it far less |
| Kyutai's "styles" are reference voices too | its sad-sympathetic voice read the apology 54 Hz lower than the default; its enunciated voice read the ticket 31% slower |
| Names need spelling help | "Domane" came out as "domain" from every model; respelling it or giving Kokoro phonemes fixed it |
| A clone copies the recording, not just the voice | cloned from a phone-line recording, Fish's output kept 0.02% of its energy above 4 kHz, against 0.75% from a clean one |

```{note}
Every number here comes from saved model outputs, so the chapter runs on a CPU in seconds. The capture script, `sim/capture_ch7.py`, ran on one rented RTX 4090. The companion notebook, `chapters/07-directing-a-voice` in [github.com/sugix/voices-of-the-future](https://github.com/sugix/voices-of-the-future), reruns Kokoro, Kyutai TTS and Fish S1-mini on a free Colab T4. Fish S2-Pro needs about 24 GB of GPU memory.
```

## The same voice for good news and bad

Here are the two lines from the opening, read by Kokoro with Chapter 5's `bf_lily` voice, and what a few simple measurements make of them:

```{code-cell} ipython3
import json
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display

sys.path.insert(0, "sim")
from wav import load  # Chapter 2's WAV reader

synth = {}
for part in ["kokoro", "kyutai", "fish1", "fish2", "fish2ref"]:
    synth |= json.load(open(f"data/ch7/synth_{part}.json"))["clips"]
heard = json.load(open("data/ch7/asr.json"))  # Parakeet-TDT's transcript of every clip
judged = json.load(open("data/ch7/judge.json"))  # pitch, pauses, loudness, emotion, voice similarity


def play(name):
    x, sr = load(f"data/ch7/listen/{name}.wav")
    display(Audio(x, rate=sr))


def words_per_second(key):
    """Words in the script (without any tag) per second of actual speech."""
    text = re.sub(r"[\(\[].*?[\)\]]", "", synth[key]["text"])
    return len(text.split()) / judged[key]["speech_s"]


def describe(key):
    j = judged[key]
    return (f"{j['f0_median']:4.0f} Hz, pitch range {j['f0_range_semitones']:4.1f} semitones, "
            f"{words_per_second(key):.1f} words/s, {j['loudness_db']:6.1f} dB")


for scene in ["good", "sorry"]:
    key = f"kokoro/{scene}_plain_clean_0"
    print(f"{scene:6s} {describe(key)}   {synth[key]['text']}")
play("kokoro_sorry_plain_clean_0")
```

Four numbers stand in for what your ear hears:

- **Median pitch** (from pYIN, a standard pitch tracker): how high the voice sits.
- **Pitch range**: how far it moves, from the 10th to the 90th percentile, in semitones. A lively voice moves more.
- **Speaking rate** in words per second of actual speech.
- **Loudness**: the median level of the speech, in dB.

For good news and bad, Kokoro sat at the same pitch and moved it by the same amount, and it read the apology a little *faster*, where a person would slow down. That's the problem this chapter tries to fix.

## Pace: the dial that works

The receptionist reads back tickets and times, and callers write them down. Slower helps. Kokoro has a `speed` argument, so I read the same read-back line at six settings:

```{code-cell} ipython3
line = "readback"
sweep = sorted((synth[k]["speed"], k) for k in synth
               if synth[k]["model"] == "kokoro" and synth[k]["scene"] == line and synth[k]["variant"] != "commas")
print(synth[sweep[0][1]]["text"], "\n")
print("speed  words/s  pause (s)  length (s)")
for speed, k in sweep:
    print(f"{speed:5.1f}  {words_per_second(k):7.2f}  {judged[k]['pause_s']:9.2f}  {synth[k]['seconds']:10.2f}")

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([s for s, _ in sweep], [words_per_second(k) for _, k in sweep], "o-")
ax.set(xlabel="Kokoro speed setting", ylabel="words per second of speech",
       title="Kokoro's speed is a real dial")
plt.tight_layout()
plt.show()

for k in ["kokoro_readback_plain_clean_0", "kokoro_readback_directed_clean_0"]:
    print(k.split("_")[2], "speed", synth[k.replace("_r", "/r", 1)]["speed"])
    play(k)
```

Speaking rate tracked the setting almost exactly: 2.5 words a second at 0.7, 4.1 at 1.3. Kokoro predicts a duration for every phoneme, and `speed` simply scales those durations, so this is as precise as a control gets. At 0.8 the read-back takes 6.8 s instead of 5.5 s, and every digit has room. I also tried the trick people reach for first, adding commas ("one, oh, four, two"): Kokoro paused no longer than before.

Now the same request in words, to the models that take written direction:

```{code-cell} ipython3
def mean(model, scene, variant, measure, ref="clean"):
    keys = [k for k, r in synth.items() if r["model"] == model and r["scene"] == scene
            and r["variant"] == variant and r.get("ref", "clean") in (ref, "ex04")]
    return np.mean([measure(k) for k in keys])


ASKS = [("fish1", "aside", "(in a hurry tone)"), ("fish2", "aside", "[quickly]"),
        ("fish2", "readback", "[slowly and clearly]"), ("kyutai", "aside", "'fast' voice"),
        ("kyutai", "readback", "'enunciated' voice")]
NAMES = {"fish1": "Fish S1-mini", "fish2": "Fish S2-Pro", "kyutai": "Kyutai TTS 1.6B", "kokoro": "Kokoro-82M"}
print(f"{'':16s} {'asked for':22s} {'plain':>8s} {'directed':>9s}  words/s")
for model, scene, ask in ASKS:
    plain, directed = (mean(model, scene, v, words_per_second) for v in ["plain", "directed"])
    print(f"{NAMES[model]:16s} {ask:22s} {plain:8.2f} {directed:9.2f}  {100 * (directed / plain - 1):+4.0f}%")
```

Written stage directions didn't move the pace. Fish S2-Pro, the most promising model here, with free-form tags and 5B parameters, read `[slowly and clearly]` 9% *faster* than the same line with no tag, and `[quickly]` no faster at all. Fish S1-mini's `(in a hurry tone)` came out 6% slower. The tags weren't ignored outright, since none were read out loud and the delivery shifted a little. But "a little, in either direction" is not a control you can build on.

Kyutai behaved differently, because it has no text tags at all. Its style comes from the voice you choose, and Kyutai's voice library has the same Expresso speaker (ex04) recorded in different styles. Her "enunciated" recording made the read-back 31% slower; her "fast" recording made the aside 15% faster. That is the first hint of the chapter's main finding.

## Emotion comes from the reference

The models that copy a voice (Fish, Kyutai, CSM in Chapter 6) learn more than timbre from the reference clip. They learn how that person was speaking. To separate the two, I gave Fish S2-Pro the same speaker, Expresso's ex04, in three recordings: default, happy and sad-sympathetic. Then I had it read the apology with and without `[sincerely apologetic, quietly]`.

```{code-cell} ipython3
rows = []
for model, ref, label in [("fish2ref", "ex04_default", "S2-Pro, neutral reference"),
                          ("fish2ref", "ex04_happy", "S2-Pro, happy reference"),
                          ("fish2ref", "ex04_sad", "S2-Pro, sad reference"),
                          ("kyutai", "default", "Kyutai, default voice"),
                          ("kyutai", "sad-sympathetic", "Kyutai, sad-sympathetic voice")]:
    for variant in ["plain", "directed"]:
        if model == "kyutai":  # Kyutai's "direction" is the style of its reference voice
            if (variant == "directed") != (ref != "default"):
                continue
            keys = [k for k, r in synth.items() if r["model"] == "kyutai" and r["scene"] == "sorry"
                    and r.get("style") == ref]
        else:
            keys = [k for k, r in synth.items() if r["model"] == model and r["scene"] == "sorry"
                    and r["ref"] == ref and r["variant"] == variant]
        f0 = np.mean([judged[k]["f0_median"] for k in keys])
        rng = np.mean([judged[k]["f0_range_semitones"] for k in keys])
        tag = "with tag" if variant == "directed" and model != "kyutai" else ""
        rows.append((f"{label} {tag}".strip(), f0, rng))

print(f"{'reading the apology':42s} {'pitch':>7s} {'range':>8s}")
for label, f0, rng in rows:
    print(f"{label:42s} {f0:5.0f} Hz {rng:5.1f} st")

fig, ax = plt.subplots(figsize=(7, 3.4))
fish_rows = {label: (f0, rng) for label, f0, rng in rows if label.startswith("S2-Pro")}
for mood, color in [("neutral", "#7f7f7f"), ("happy", "#4a7fb0"), ("sad", "#c0504d")]:
    plain = fish_rows[f"S2-Pro, {mood} reference"]
    tagged = fish_rows[f"S2-Pro, {mood} reference with tag"]
    ax.scatter(*plain, s=50, color=color)
    ax.annotate("", xy=tagged, xytext=plain, arrowprops=dict(arrowstyle="->", color=color))
    ax.annotate(f"{mood} reference", plain, textcoords="offset points", xytext=(-10, 8), fontsize=8, color=color)
ax.set(xlabel="median pitch (Hz)", ylabel="pitch range (semitones)", xlim=(190, 275), ylim=(2, 11),
       title="S2-Pro's apology: the dot is the reference, the arrow is the tag")
plt.tight_layout()
plt.show()

for name in ["fish2ref_sorry_plain_ex04_happy_0", "fish2ref_sorry_plain_ex04_sad_0", "kyutai_sorry_directed_clean_0"]:
    print(name.replace("fish2ref_", "S2-Pro ").replace("_0", "").replace("_", " "))
    play(name)
```

The reference decided the delivery. From the sad recording, S2-Pro's apology dropped to about 205 Hz and nearly flat (about 3 semitones of movement), which is how people sound when they are sorry. From the neutral and happy recordings, it stayed at 240 to 260 Hz with twice the movement. Adding `[sincerely apologetic, quietly]` moved each of these by less than the gap between references, and with the neutral reference it moved the pitch *up*. Kyutai showed the same thing: swapping ex04's default voice for her sad-sympathetic one lowered the apology by 54 Hz.

This changes how you direct a voice. With a cloning model, **you don't direct the reading, you cast it**. Record, or choose, a reference clip for each delivery you need, in the voice you want: a warm one for good news, a quiet, slow one for apologies, a clear, measured one for read-backs. Then pick the clip per reply. The tags are worth a try on top, but the clip does the work.

```{warning}
I also ran an off-the-shelf emotion classifier (wav2vec2 trained on IEMOCAP's four acted emotions) over every clip. It was no help: it almost never said "sad", even for the sad-reference apologies, and it heard "happy" most strongly in good news cloned from the *sad* recording. Classifiers trained on acted speech don't transfer to synthetic voices. Pitch, range and pace, plus your own ears, were more trustworthy judges.
```

## Saying names right

Chapter 6 found Kokoro saying "OCK-a-fuh" for Okafor. Every receptionist meets names and products the model has never seen. I tried four: Okafor, Aoife and Siobhan (two Irish names whose spelling hides the sound), and the Trek Domane (said "doh-MAH-nee"). Each model read each name as written, then with help:

- Kokoro got phonemes inline, `[Domane](/dəmˈɑːni/)`, in its own British symbols.
- Fish and Kyutai, which take no phonemes, got a respelling: "Doh-mah-nee", "Eefa", "Shiv-awn".

```{code-cell} ipython3
print(f"{'':16s} {'name':8s} {'as written':26s} {'with help':26s}")
for model in ["kokoro", "fish1", "fish2", "kyutai"]:
    helped = "phonemes" if model == "kokoro" else "respelled"
    for name in ["Okafor", "Aoife", "Siobhan", "Domane"]:
        plain, fixed = (heard[f"{model}/name_{name}_{v}_0"] for v in ["spelled", helped])
        print(f"{NAMES[model]:16s} {name:8s} {plain[:26]:26s} {fixed[:26]:26s}")
print()
for v in ["spelled", "phonemes"]:
    print("Kokoro, Okafor", v, ":", synth[f"kokoro/name_Okafor_{v}_0"]["phonemes"])
    play(f"kokoro_name_Okafor_{v}_0")
```

These are Parakeet's transcripts, so read them as "what a listener heard" rather than as scores; no recogniser spells Aoife the way the sound goes. Even so, the pattern is clear:

- **"Domane" came out as "domain" from every model.** With help it became "Domani", "Demani" or "dominie": the stress moved to where it belongs.
- **Kokoro's phonemes are exact.** The phonemes it spoke are the ones I gave it, character for character, so a lexicon of overrides is fully predictable.
- **Respelling works, but it is a guess.** Fish S1-mini turned "Shiv-awn" into "Chivon", which is right, but S2-Pro split it into "Shiva On". Hyphens and capitals are hints, not rules. Listen to every respelling once before it goes into the lexicon.
- **Some names were already fine.** Kokoro, Kyutai and S2-Pro all said Siobhan correctly as written, and Kokoro knew Aoife. Check before you fix.

## A clone copies the recording, not just the voice

Halfwheel might want the receptionist to sound like a real member of staff, and the easiest recording to get of them is a phone call. So I cloned the same speaker from two versions of one recording: the clean studio clip, and the same clip through Chapter 2's phone line (8 kHz, 300 to 3,400 Hz, μ-law).

```{code-cell} ipython3
print(f"{'':14s} {'reference':10s} {'energy above 4 kHz':>19s} {'similarity to the speaker':>27s}")
for model in ["fish1", "fish2"]:
    for ref in ["clean", "phone"]:
        keys = [k for k, r in synth.items() if r["model"] == model and r.get("ref") == ref and r["variant"] == "plain"
                and not r["scene"].startswith("name_")]
        above = np.mean([judged[k]["above_4k"] for k in keys])
        sim = np.mean([judged[k]["sim_p228"] for k in keys])
        print(f"{NAMES[model]:14s} {ref:10s} {100 * above:18.2f}% {sim:27.3f}")
print("\nthe reference, over the phone line:")
play("ref_p228_phone")
print("S2-Pro cloned from it:")
play("fish2_good_plain_phone_0")
```

Both Fish models reproduced the phone line along with the voice. Cloned from the clean clip, about 0.75% of the energy of their speech was above 4 kHz, the range that carries "s", "f" and "th". Cloned from the phone recording, it was 0.01 to 0.02%: effectively none. The model can't know which parts of the reference were the speaker and which were the line, so it copies both. The speaker-similarity score (cosine similarity of WavLM speaker embeddings, where Kokoro's unrelated voice still scores about 0.88) barely moved, which is a reminder that "sounds like the same person" and "sounds good" are different measurements. If you clone, record the reference properly: quiet room, decent microphone, a sentence or two in each delivery you need.

How fast are they? Run through Fish's reference engine without compilation, S1-mini made audio at half real time on the 4090 and S2-Pro at a fifth. Fish reports S2-Pro at about five times faster than real time on an H200 with its SGLang-based streaming engine. For a live phone call, serve it the way Fish does, or use it to pre-render the receptionist's stock phrases.

```{admonition} Licences, briefly
Kokoro is Apache 2.0 and Kyutai TTS is CC BY 4.0. Fish's open weights are for non-commercial use: S1-mini under CC BY-NC-SA 4.0, S2-Pro under the Fish Audio Research License. Expresso, the source of Kyutai's style voices, is CC BY-NC 4.0. For a commercial receptionist, record your own references, with consent.
```

## Halfwheel's voice direction

Putting the chapter together, the receptionist gets three controls:

1. **Pace, from the engine.** Read-backs of tickets, times and phone numbers at Kokoro's speed 0.8, or with a slower reference clip on a cloning model; everything else at normal speed.
2. **Pronunciation, from a lexicon.** Chapter 6's `speakable()` gains entries for the shop's awkward names: phonemes for Kokoro, respellings for the rest, each one checked by ear.
3. **Delivery, from casting.** For a cloning model, a few reference clips of the same voice: warm for good news, quiet and slow for apologies and complaints, clear for read-backs. The language model already knows which kind of reply it is writing, so it tags each reply with one of them, and the speaking step picks the clip.

Chapter 8 joins listening, thinking and speaking into a receptionist that answers the phone.

## When it breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| Every reply sounds equally cheerful | One reference clip, or a model with a single voice | Cast several reference clips by delivery and pick per reply |
| A tag like `[slowly]` changes nothing, or the opposite | Written direction is a weak hint, not a control | Use the engine's speed setting, or a slower reference clip |
| A tag is read out loud | Wrong syntax for that model (Fish S1 uses `(...)`, S2-Pro uses `[...]`) | Check the model's tag format; listen to one clip per tag |
| A name is wrong every time | Spelling that hides the sound | Phonemes (Kokoro) or a respelling in the lexicon |
| A respelling makes it worse | The model split or re-read the hint | Try another respelling; check every entry by ear |
| The cloned voice sounds muffled or "on the phone" | The reference was a phone recording | Record references in a quiet room at full bandwidth |
| An emotion classifier disagrees with your ears | Trained on acted human speech | Measure pitch, range and pace; trust a listening test |

```{tip}
**Takeaways**
- Separate *what* is said from *how*. Pace and pronunciation have real dials: the engine's speed, and phonemes or respellings in a lexicon.
- Written stage directions are hints. In my tests they moved the delivery a little, sometimes the wrong way.
- With a cloning model, delivery comes from the reference clip. Cast the voice: one clip per kind of reply, same speaker.
- A clone copies the recording, including its noise and bandwidth. Record references properly.
- Measure delivery with pitch, pitch range, pace and loudness, and listen. Off-the-shelf emotion classifiers don't transfer to synthetic speech.
```

## Try it yourself

The companion notebook, `chapters/07-directing-a-voice`, has the saved outputs, every cell above and the capture script.

1. **Cast your own voice.** Record yourself reading one sentence warmly, one apologetically and one slowly, ten seconds each, in a quiet room. Clone each with Fish S1-mini and read the same apology. Measure pitch and range with the `describe()` helper.
2. **Tag placement.** Fish's documentation shows tags inline as well as at the start. Put `[pause]` after "I'm really sorry." and measure the pause.
3. **Your own lexicon.** List five names or products your callers say. Which does Kokoro already get right? Write phonemes for the rest and check the phonemes it returns.
4. **Phone-line references.** Clone from the phone-line reference, then from the clean one, and listen to both on headphones. Can you hear the 4 kHz cut-off?

## Next

The receptionist can now hear a caller, know when they have finished, and answer in the right words, at the right pace, in the right tone. Each piece works on its own. Chapter 8 puts them together into one receptionist that answers Halfwheel's phone, and measures the whole thing from the caller's first word to the receptionist's reply.
