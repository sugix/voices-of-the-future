---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
date: "2027-02-01"
feed: false
description: Every symptom and fix from the book in one place, and every model the book ran, with its size, licence and where it runs.
---

# Field guide

This appendix is for when something is already wrong. The first part collects every chapter's "When it breaks" table into one, so you can search for the symptom you're seeing. The second lists every model the book ran.

## Symptoms and fixes

This table is built from the chapters themselves each time the book is built, so it always matches them.

```{code-cell} ipython3
:tags: [remove-input]
import re
from pathlib import Path
import html

from IPython.display import HTML, display


def inline(text):
    """The little markdown these tables use: `code` and *emphasis*."""
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)


CHAPTERS = ["audio-for-ml", "asr-decoders", "choosing-asr", "listening-live", "how-tts-works", "directing-a-voice",
            "building-the-receptionist", "screening-calls", "speech-to-speech", "evaluating-voice-agents",
            "real-phone-line"]
rows = []
for name in CHAPTERS:
    text = Path(f"{name}.md").read_text()
    title = re.search(r"^# (.+)$", text, re.M)[1]
    section = re.search(r"^## When it breaks\n(.*?)(?=^```|^## )", text, re.M | re.S)
    if not section:
        continue
    for line in section[1].splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] not in ("Symptom", "") and not set(cells[0]) <= set("-: "):
            rows.append([inline(c) for c in cells] + [f'<a href="{name}.html">{html.escape(title)}</a>'])
head = "".join(f"<th>{h}</th>" for h in ["Symptom", "Likely cause", "Fix", "Chapter"])
body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
display(HTML(f'<table class="table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'))
```

## Models used in this book

| Model | What it does | Size | Licence | Runs on | Chapters |
|---|---|---|---|---|---|
| Silero VAD | voice activity detection | about 2 MB | MIT | CPU | 2, 5, 8 |
| Parakeet-TDT 0.6B v2 | speech recognition (transducer) | 0.6B | CC BY 4.0 | GPU; slowly on CPU | 2 to 12 |
| Parakeet-CTC 0.6B | speech recognition (CTC) | 0.6B | CC BY 4.0 | GPU | 3, 4 |
| Parakeet realtime EOU 120M | streaming recognition with end of utterance | 120M | NVIDIA Open Model Licence | GPU | 5 |
| Whisper large-v3-turbo | speech recognition (LLM-style decoder) | 0.8B | MIT | GPU | 3, 4 |
| Qwen3-ASR | speech recognition (LLM decoder) | 0.6B, 1.7B | Apache 2.0 | GPU | 3, 4 |
| Kyutai STT | streaming speech recognition | 1B, 2.6B | CC BY 4.0 | GPU | 4, 5 |
| Kokoro-82M | text to speech (phonemes and a vocoder) | 82M | Apache 2.0 | CPU or GPU | 5 to 8, 11, 12 |
| Pocket TTS | text to speech (continuous latents) | 100M | CC BY 4.0 | CPU | 6, 8, 12 |
| Kyutai TTS 1.6B | streaming text to speech | 1.6B | CC BY 4.0 | GPU | 6, 7 |
| CSM-1B | conversational text to speech | 1B | Apache 2.0 | GPU | 6, 10 |
| Fish Audio S1-mini | text to speech with emotion tags | 0.5B | CC BY-NC-SA 4.0 | GPU | 7 |
| Fish Audio S2-Pro | text to speech with free-form direction | 5B | Fish Audio Research Licence | GPU (24 GB) | 7 |
| Moshi | full-duplex speech to speech | 7B | CC BY 4.0 | GPU (24 GB) | 10 |
| Qwen3 1.7B, 4B, 8B | the receptionist's language model, and judges | 1.7B to 8B | Apache 2.0 | GPU (8B: more than 16 GB) | 8, 9, 11, 12 |
| wav2vec2 (SUPERB emotion) | speech emotion classification | 95M | Apache 2.0 | GPU or CPU | 7 |
| WavLM (speaker verification) | voice similarity | 95M | not stated; see the model card | GPU or CPU | 7 |

Licences change, and some voices and datasets carry their own terms (Chapters 6 and 7 note the Expresso voices are non-commercial). Check the model card before you ship.
