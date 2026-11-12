# Voices of the Future: companion notebooks

Runnable notebooks for [*Voices of the Future*](https://books.sugiv.fyi/voice_applications/intro.html), a hands-on
book about building with modern voice AI models. Across the book you build one project: a voice receptionist for
**Halfwheel Cycles**, a fictional bike repair shop.

## Two ways to run each chapter

1. **From the saved results (any laptop, or Colab without a GPU).** Open a chapter's notebook and run it top to
   bottom: it reproduces every table and figure in the chapter from the saved model outputs, in seconds.
2. **Rerun the models (a GPU).** The cells at the end of each notebook rerun the models that produced those results.
   Each step builds its own Python environment with `uv` the first time, so libraries with conflicting versions
   never meet. Set `VOTF_QUICK=2` for a quick check first.

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
