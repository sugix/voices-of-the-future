# Voices of the Future: companion notebooks

Runnable notebooks for [*Voices of the Future*](https://books.sugiv.fyi/voice_applications/intro.html), a hands-on
book about building with modern voice AI models. Across the book you build one project: a voice receptionist for
**Halfwheel Cycles**, a fictional bike repair shop.

## Chapters

| | Chapter | What you build and measure | Run it | Rerun the models on |
|---|---|---|---|---|
| 1 | [The voice stack](https://books.sugiv.fyi/voice_applications/the-voice-stack.html) | The whole voice-agent stack on a few pages, from the phone line to the evaluation plan, with the model chosen and the number measured at every layer, and a map of the book for different readers. | no code: the map of the book |  |
| 2 | [Audio for ML engineers](https://books.sugiv.fyi/voice_applications/audio-for-ml.html) | What a phone call looks like to a speech model, built from scratch on one Halfwheel call, and what common audio mistakes really cost. | [notebook](chapters/02-audio-for-ml.ipynb) · [Colab](https://colab.research.google.com/github/sugix/voices-of-the-future/blob/main/chapters/02-audio-for-ml.ipynb) | a CPU |
| 3 | [How speech recognition decoders read audio](https://books.sugiv.fyi/voice_applications/asr-decoders.html) | CTC, transducers and LLM decoders, built from scratch in numpy, and what each one means for speed, streaming and getting ticket numbers right. | [notebook](chapters/03-asr-decoders.ipynb) · [Colab](https://colab.research.google.com/github/sugix/voices-of-the-future/blob/main/chapters/03-asr-decoders.ipynb) | a free Colab T4 |
| 4 | [Choosing speech recognition for your callers](https://books.sugiv.fyi/voice_applications/choosing-asr.html) | Six speech recognition models, 30 bike-shop calls and 80 real recordings, scored on the words a business actually depends on. | [notebook](chapters/04-choosing-asr.ipynb) · [Colab](https://colab.research.google.com/github/sugix/voices-of-the-future/blob/main/chapters/04-choosing-asr.ipynb) | a free Colab T4 |
| 5 | [Listening live](https://books.sugiv.fyi/voice_applications/listening-live.html) | Streaming recognition, knowing when a caller has finished, and handling interruptions, measured on hesitant callers and a receptionist that hears its own echo. | [notebook](chapters/05-listening-live.ipynb) · [Colab](https://colab.research.google.com/github/sugix/voices-of-the-future/blob/main/chapters/05-listening-live.ipynb) | a free Colab T4 |
| 6 | [How modern TTS works](https://books.sugiv.fyi/voice_applications/how-tts-works.html) | Four open text-to-speech models read Halfwheel's receptionist script. | [notebook](chapters/06-how-tts-works.ipynb) · [Colab](https://colab.research.google.com/github/sugix/voices-of-the-future/blob/main/chapters/06-how-tts-works.ipynb) | a free Colab T4 |
| 7 | [Directing a voice](https://books.sugiv.fyi/voice_applications/directing-a-voice.html) | Pace, pronunciation, emotion and cloned voices, measured across Kokoro, Kyutai TTS and Fish Audio's S1-mini and S2-Pro. | [notebook](chapters/07-directing-a-voice.ipynb) · [Colab](https://colab.research.google.com/github/sugix/voices-of-the-future/blob/main/chapters/07-directing-a-voice.ipynb) | a T4; Fish S2-Pro needs 24 GB |
| 8 | [Building the receptionist](https://books.sugiv.fyi/voice_applications/building-the-receptionist.html) | Chapters 2 to 7 joined into one receptionist with a small language model and four tools, run against sixteen callers who talk back. | [notebook](chapters/08-building-the-receptionist.ipynb) · [Colab](https://colab.research.google.com/github/sugix/voices-of-the-future/blob/main/chapters/08-building-the-receptionist.ipynb) | a T4; the Qwen3-8B runs need more than 16 GB |
| 9 | [Screening calls](https://books.sugiv.fyi/voice_applications/screening-calls.html) | Real robocalls against Halfwheel's genuine callers, including the ones that sound like scams. | [notebook](chapters/09-screening-calls.ipynb) · [Colab](https://colab.research.google.com/github/sugix/voices-of-the-future/blob/main/chapters/09-screening-calls.ipynb) | a free Colab T4 |
| 10 | [From Spirit LM to Moshi and CSM](https://books.sugiv.fyi/voice_applications/speech-to-speech.html) | Models that take speech in and give speech out, from Spirit LM to Moshi and CSM. | [notebook](chapters/10-speech-to-speech.ipynb) · [Colab](https://colab.research.google.com/github/sugix/voices-of-the-future/blob/main/chapters/10-speech-to-speech.ipynb) | 24 GB for Moshi; CSM fits a T4 |
| 11 | [Evaluating voice agents](https://books.sugiv.fyi/voice_applications/evaluating-voice-agents.html) | How much a voice agent's score is worth. | [notebook](chapters/11-evaluating-voice-agents.ipynb) · [Colab](https://colab.research.google.com/github/sugix/voices-of-the-future/blob/main/chapters/11-evaluating-voice-agents.ipynb) | a T4; the Qwen3-8B judge needs more than 16 GB |
| 12 | [Putting it on a real phone line](https://books.sugiv.fyi/voice_applications/real-phone-line.html) | Connecting the receptionist to a real phone line. | [notebook](chapters/12-real-phone-line.ipynb) · [Colab](https://colab.research.google.com/github/sugix/voices-of-the-future/blob/main/chapters/12-real-phone-line.ipynb) | a 16 GB GPU; vLLM not tested on a T4 |
| | [Field guide](https://books.sugiv.fyi/voice_applications/field-guide.html) | Every model, its licence and what it needs; every chapter's "When it breaks" table | | |

## Running them

```bash
git clone https://github.com/sugix/voices-of-the-future && cd voices-of-the-future
python -m venv .venv && . .venv/bin/activate     # Python 3.12
pip install -r requirements/results.txt jupyterlab
jupyter lab chapters/
```

Open a chapter and run it top to bottom; the first cell finds the book's folder, so it works from anywhere in the
clone. In Google Colab, use the Colab link in the table: the first cell clones this repository for you. From the
saved results every notebook runs in seconds and needs no GPU or downloads.

## Two ways to run each chapter

1. **From the saved results (any laptop, or Colab without a GPU).** Open a chapter's notebook and run it top to
   bottom: it reproduces every table and figure in the chapter from the saved model outputs, in seconds.
2. **Rerun the models (a GPU).** The cells at the end of each notebook rerun the models that produced those results.
   Each step builds its own Python environment with `uv` the first time, so libraries with conflicting versions
   never meet. Run the quick check first: it works on a scratch copy and leaves the saved results alone.

In Google Colab, open a notebook from this repository and choose Runtime, then Change runtime type, then T4 GPU
for the rerun cells.

## What fits a free Colab T4 (16 GB, no bfloat16)

I ran every model step on a 16 GB GPU in float16, the T4's limits. Everything fits except:

| Step | Needs |
|---|---|
| Chapter 7: Fish Audio S2-Pro | about 24 GB |
| Chapter 8: the Qwen3-8B receptionist runs | more than 16 GB in float16 |
| Chapter 10: Moshi | 24 GB (neither the float16 nor the 8-bit PyTorch build fits 16 GB) |
| Chapter 11: the Qwen3-8B judge | more than 16 GB in float16 |
| Chapter 12: the vLLM load test | ran on 16 GB, but not tested on a T4's older chip |

Chapter 7's Fish S1-mini weights are gated: accept the licence at huggingface.co/fishaudio/s1-mini and log in.

## Layout

- `chapters/` one notebook per chapter (2 to 12; Chapter 1 has no code)
- `voice/` the book's chapters, simulators (`sim/`), saved results (`data/`) and `tools/stack.sh`
- `requirements/results.txt` what the saved-results half needs

Code is MIT-licensed. Models, voices and datasets keep their own licences, listed in the book's field guide.
