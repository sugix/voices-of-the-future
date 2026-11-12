"""Two switches every capture script honours, for smaller GPUs and quick checks.

  VOTF_QUICK=N   process only the first N items of each loop (a smoke test, not a result)
  VOTF_FP16=1    use float16 even where bfloat16 is available, as on a Colab T4 (which has no bfloat16)
"""

import os
import tempfile
from pathlib import Path

SCRATCH = Path(os.environ.get("VOTF_TMP", Path(tempfile.gettempdir()) / "votf"))  # temporary audio, not results
SCRATCH.mkdir(parents=True, exist_ok=True)


def quick(items):
    n = int(os.environ.get("VOTF_QUICK", "0") or 0)
    return list(items)[:n] if n else items


def half():
    """bfloat16 where the GPU supports it, otherwise float16."""
    import torch
    if os.environ.get("VOTF_FP16") or torch.cuda.get_device_capability()[0] < 8:  # T4 (sm_75) only emulates bf16
        return torch.float16
    return torch.bfloat16


def existing(clips, root="."):
    """(key, row) for clips whose audio is on disk. A model you didn't rerun (say, one that needs a bigger GPU)
    keeps its saved results, but its audio isn't there to score again, so it's skipped with a note."""
    missing = [k for k, r in clips.items() if not (Path(root) / r["wav"]).exists()]
    if missing:
        print(f"skipping {len(missing)} clips with no audio on disk (e.g. {missing[0]}); rerun that model to include them")
    return [(k, r) for k, r in clips.items() if k not in set(missing)]


def merge_json(path, new):
    """Write new results over the saved ones, keeping saved entries for anything not rerun."""
    import json
    path = Path(path)
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(new)
    path.write_text(json.dumps(old, indent=0))


def fish_paths(n):
    """Code and weights for Fish S1-mini (n=1) or S2-Pro (n=2): from FISH<n>_CODE / FISH<n>_CKPT if set,
    otherwise downloaded once into SCRATCH. S1-mini's weights are gated: accept the licence on Hugging Face
    and log in (huggingface-cli login) first."""
    import subprocess
    from huggingface_hub import snapshot_download
    code = Path(os.environ.get(f"FISH{n}_CODE", SCRATCH / f"fish{n}-code"))
    ckpt = Path(os.environ.get(f"FISH{n}_CKPT", SCRATCH / f"fish{n}-weights"))
    if not code.exists():
        if n == 1:
            snapshot_download("fishaudio/openaudio-s1-mini", repo_type="space", local_dir=code)
        else:
            subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", "v2.0.0-beta",
                            "https://github.com/fishaudio/fish-speech", str(code)], check=True)
    if not ckpt.exists():
        snapshot_download("fishaudio/s1-mini" if n == 1 else "fishaudio/s2-pro", local_dir=ckpt)
    return str(code), str(ckpt)
