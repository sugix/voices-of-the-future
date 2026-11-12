"""Build the companion repo's notebooks from the book's chapters.

  python3 voice_applications/tools/make_companion.py ~/vscode_home/voices-of-the-future

Copies the git-tracked chapter sources, sim/ and data/ into <repo>/voice/, and writes one notebook per
chapter to <repo>/chapters/: section headings and every code cell from the chapter, in order, after a setup
cell that works both in Google Colab and in a local clone. The prose stays in the book; each notebook links
to its chapter. Plain Python only (no Jupyter needed to build).
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parents[1]
SITE = "https://books.sugiv.fyi/voice_applications"
REPO = "https://github.com/sugix/voices-of-the-future"
CHAPTERS = ["audio-for-ml", "asr-decoders", "choosing-asr", "listening-live", "how-tts-works", "directing-a-voice",
            "building-the-receptionist", "screening-calls", "speech-to-speech", "evaluating-voice-agents",
            "real-phone-line"]
RESULTS_REQUIREMENTS = "numpy\nscipy\npandas\nmatplotlib\nipython\nsoundfile\n"

# Rerun steps per chapter: (stack, command, note). Each stack is its own environment (tools/stack.sh), so
# stacks with conflicting pins never meet. Steps run in order; later ones read what earlier ones wrote.
CALLS = [("tts", "sim/halfwheel_calls.py", "Halfwheel's calls: Kokoro reads them, Chapter 2's phone line carries them.")]
LIBRI = [("nemo", "sim/librispeech_subset.py", "80 LibriSpeech recordings (downloads about 700 MB).")]
RERUN = {
    "audio-for-ml": CALLS + [("nemo", "sim/capture_ch2.py", "")],
    "asr-decoders": CALLS + [("nemo", "sim/capture_ch3.py", "")],
    "choosing-asr": CALLS + LIBRI + [("nemo", "sim/capture_ch4.py", "Six recognisers on 110 files; about 15 minutes on a T4.")],
    "listening-live": CALLS + LIBRI + [("tts", "sim/turn_calls.py", ""), ("nemo", "sim/capture_ch5.py --part nemo", ""),
                                       ("moshi", "sim/capture_ch5.py --part kyutai", ""), ("tts", "sim/aec_ch5.py --synth", ""),
                                       ("nemo", "sim/aec_ch5.py --apply", "")],
    "how-tts-works": LIBRI + [("nemo", "sim/capture_ch6.py --part voice", ""), ("tts", "sim/capture_ch6.py --part kokoro", ""),
                              ("tts", "sim/capture_ch6.py --part pocket", ""), ("moshi", "sim/capture_ch6.py --part kyutai", ""),
                              ("hf", "sim/capture_ch6.py --part csm", ""), ("nemo", "sim/capture_ch6.py --part asr", "")],
    "directing-a-voice": [("nemo", "sim/capture_ch6.py --part voice", "The reference voice from Chapter 6."),
                          ("tts", "sim/capture_ch7.py --part kokoro", ""), ("moshi", "sim/capture_ch7.py --part kyutai", ""),
                          ("fish1", "sim/capture_ch7.py --part fish1",
                           "Fish S1-mini: accept its licence at huggingface.co/fishaudio/s1-mini and log in first."),
                          ("fish2", "sim/capture_ch7.py --part fish2",
                           "Fish S2-Pro needs about 24 GB of GPU memory; it doesn't fit a 16 GB T4. Skip it there."),
                          ("nemo", "sim/capture_ch7.py --part refs", ""), ("nemo", "sim/capture_ch7.py --part asr", ""),
                          ("hf", "sim/capture_ch7.py --part judge", "")],
    "building-the-receptionist": [("tts", "sim/capture_ch8.py --part voice", ""), ("nemo", "sim/capture_ch8.py --part hear", ""),
                                  ("hf", "sim/capture_ch8.py --part talk", "Qwen3-8B doesn't fit a 16 GB T4 in fp16; its runs fail there."),
                                  ("tts", "sim/capture_ch8.py --part speak", "")],
    "screening-calls": [("nemo", "sim/capture_ch9.py --part fetch", "Downloads 150 robocalls (public domain) from the NCSU dataset."),
                        ("tts", "sim/capture_ch9.py --part voice", ""), ("nemo", "sim/capture_ch9.py --part hear", ""),
                        ("hf", "sim/capture_ch9.py --part judge", "")],
    "speech-to-speech": [("moshi", "sim/capture_ch10.py --part moshi", "Moshi needs a 24 GB GPU: it does not fit a 16 GB T4, in float16 or with its 8-bit build."),
                         ("nemo", "sim/capture_ch10.py --part refs", ""), ("hf", "sim/capture_ch10.py --part csm", "")],
    "evaluating-voice-agents": [("tts", "sim/capture_ch11.py --part voice", ""), ("nemo", "sim/capture_ch11.py --part hear", ""),
                                ("hf", "sim/capture_ch11.py --part talk", ""),
                                ("hf", "sim/capture_ch11.py --part judge", "The judge is Qwen3-8B, which needs more than a T4's 16 GB."),
                                ("hf", "sim/capture_ch11.py --part judge2", "The same 8B judge, given the expected behaviour.")],
    "real-phone-line": [("vllm", "sim/capture_ch12.py --part replay", ""), ("vllm", "sim/capture_ch12.py --part llm", "Tested on 16 GB in float16, but not on a T4 itself: recent vLLM releases may not support its older chip."),
                        ("nemo", "sim/capture_ch12.py --part asr", ""), ("tts", "sim/capture_ch12.py --part tts", "")],
}
RERUN_INTRO = """## Rerun the models

Everything above used the saved results. The cells below rerun the models that produced them, on a GPU (in Colab:
Runtime, then Change runtime type, then T4 GPU). Each step builds the software it needs in its own environment the
first time, which takes a few minutes, then runs. Set `VOTF_QUICK=2` first for a two-item check that everything
works; leave it unset for a full rerun, which overwrites the saved results. On GPUs without bfloat16, such as the
T4, the scripts use float16 automatically."""

SETUP = f'''# Run this cell first. In Google Colab it fetches the companion repo; anywhere else, run the notebook
# from a clone of it. Either way it moves into the book's folder and installs what the chapter needs.
import os
import pathlib
import sys

if "google.colab" in sys.modules and not pathlib.Path("/content/voices-of-the-future").exists():
    !git clone -q --depth 1 {REPO} /content/voices-of-the-future
candidates = [pathlib.Path.cwd(), *pathlib.Path.cwd().parents, pathlib.Path("/content/voices-of-the-future")]
root = next(p for p in candidates if (p / "voice" / "sim").is_dir())
os.chdir(root / "voice")
%pip install -q -r ../requirements/results.txt'''


README = """# Voices of the Future: companion notebooks

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
"""


def cell(kind, text, tags=None):
    c = {"cell_type": kind, "metadata": {"tags": tags} if tags else {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c |= {"execution_count": None, "outputs": []}
    return c


def notebook(number, slug):
    src = (BOOK / f"{slug}.md").read_text()
    title = re.search(r"^# (.+)$", src, re.M)[1]
    cells = [cell("markdown", f"# Chapter {number}: {title}\n\nThis notebook runs every piece of code in the chapter, "
                              f"from the saved results, in order. Read the chapter alongside it: "
                              f"[{title}]({SITE}/{slug}.html).\n"),
             cell("code", SETUP)]
    body = re.sub(r"^---\n.*?\n---\n", "", src, count=1, flags=re.S)
    fence = "`" * 3
    pattern = re.compile(r"^(#{2,3} [^\n]+)$|^" + fence + r"\{code-cell\} ipython3\n(.*?)^" + fence + r"$", re.M | re.S)
    for m in pattern.finditer(body):
        if m.group(1):
            cells.append(cell("markdown", m.group(1)))
            continue
        code = m.group(2)
        options = re.match(r"((?::[a-z-]+:.*\n)+)", code)
        if options and "remove-cell" in options.group(1):
            continue  # the book's hidden cells feed numbers into its prose; a notebook doesn't need them
        if options:
            code = code[options.end():]
        cells.append(cell("code", code.rstrip("\n")))
    while cells and cells[-1]["cell_type"] == "markdown":
        cells.pop()  # a heading with no code after it
    steps = RERUN.get(slug, [])
    if steps:
        tag = ["rerun"]  # model steps: needs a GPU; skipped by the saved-results check
        cells.append(cell("markdown", RERUN_INTRO, tag))
        cells.append(cell("code", "import os\nos.environ[\"VOTF_QUICK\"] = \"2\"  # remove this line for a full rerun", tag))
        for stack, command, note in steps:
            if note:
                cells.append(cell("markdown", note, tag))
            cells.append(cell("code", f"!bash tools/stack.sh {stack}\n!~/.votf/envs/{stack}/bin/python {command}", tag))
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"}, "colab": {"provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}


def main(dest):
    dest = Path(dest).expanduser()
    voice = dest / "voice"
    if voice.exists():
        shutil.rmtree(voice)
    tracked = subprocess.run(["git", "ls-files", "data", "sim", "tools", ":(glob)*.md", "references.bib"], cwd=BOOK,
                             capture_output=True, text=True, check=True).stdout.split()
    for rel in tracked:
        (voice / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BOOK / rel, voice / rel)
    (dest / "requirements").mkdir(exist_ok=True)
    (dest / "requirements" / "results.txt").write_text(RESULTS_REQUIREMENTS)
    out = dest / "chapters"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.ipynb"):
        old.unlink()
    for number, slug in enumerate(CHAPTERS, start=2):
        nb = notebook(number, slug)
        (out / f"{number:02d}-{slug}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"{number:02d}-{slug}.ipynb", sum(c["cell_type"] == "code" for c in nb["cells"]) - 1, "code cells")
    print(len(tracked), "files copied into", voice)
    (dest / "README.md").write_text(README)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "~/vscode_home/voices-of-the-future")
