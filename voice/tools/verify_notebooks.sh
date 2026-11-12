#!/bin/bash
# Run every chapter notebook from the saved results, headless, in a fresh environment, skipping the GPU rerun cells.
#   bash voice/tools/verify_notebooks.sh            (from the companion repo's root)
# Prints PASS or FAIL per notebook; executed copies and logs go to $OUT (default /tmp/votf-notebooks).
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-/tmp/votf-notebooks}
mkdir -p "$OUT"
command -v uv >/dev/null || pip install -q uv
[ -x "$OUT/.venv/bin/python" ] || uv venv -q --python 3.12 "$OUT/.venv"
uv pip install -q --python "$OUT/.venv/bin/python" jupyter nbconvert ipykernel pip
. "$OUT/.venv/bin/activate"
cd "$root/chapters"
for nb in *.ipynb; do
  if jupyter nbconvert --to notebook --execute --TagRemovePreprocessor.enabled=True \
       --TagRemovePreprocessor.remove_cell_tags=rerun --ExecutePreprocessor.timeout=900 \
       --output-dir "$OUT" "$nb" > "$OUT/${nb%.ipynb}.log" 2>&1; then
    echo "PASS $nb"
  else
    echo "FAIL $nb (see $OUT/${nb%.ipynb}.log)"
  fi
done
