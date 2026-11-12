"""Quick check of capture_ch4.json using the chapter's own scoring code. Run from voice_applications/."""
import contextlib
import io
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
src = pathlib.Path("choosing-asr.md").read_text()
cells = re.findall(r"```\{code-cell\} ipython3\n(.*?)```", src, re.S)
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    for c in cells[:4]:
        exec(c, ns)
normalize, edit_ops = ns["normalize"], ns["edit_ops"]

cap = json.load(open("data/ch4/capture_ch4.json"))
refs = {}
for root, s in [("data/calls", "halfwheel"), ("data/librispeech", "librispeech")]:
    for line in open(f"{root}/manifest.jsonl"):
        r = json.loads(line)
        refs[(s, r["id"])] = r

for name, res in cap["results"].items():
    if "files" not in res:
        print(name, "ERROR")
        continue
    agg, ent_by, tot_s, tot_a = {}, {}, 0.0, 0.0
    for key, r in res["files"].items():
        s, cond, uid = key.split("/")
        ref = refs[(s, uid)]
        R, H = normalize(ref["written"]).split(), normalize(r["text"]).split()
        a = agg.setdefault((s, cond), [0, 0])
        a[0] += sum(edit_ops(R, H).values())
        a[1] += len(R)
        tot_s += r["s"]
        tot_a += r["audio_s"]
        h = f" {normalize(r['text'])} "
        for en in ref.get("entities", []):
            b = ent_by.setdefault((cond, en["type"]), [0, 0])
            b[0] += f" {normalize(en['written'])} " in h
            b[1] += 1
    w = " ".join(f"{s[:5]}/{c}={a[0] / a[1]:.1%}" for (s, c), a in sorted(agg.items()))
    ents = " ".join(f"{c[:2]}/{t}={v[0]}/{v[1]}" for (c, t), v in sorted(ent_by.items()))
    print(f"{name:40} {w}\n{'':40} {ents}  RTFx={tot_a / tot_s:.0f} peak={res['peak_gb']}GB")
