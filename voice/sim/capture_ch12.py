"""Chapter 12 capture: how many calls one GPU can carry.

  python sim/capture_ch12.py --part replay  # rebuild every model request Chapter 8's receptionist made (any venv)
  python sim/capture_ch12.py --part llm     # Qwen3-4B on vLLM: those requests at 1 to 64 in flight (vllm venv)
  python sim/capture_ch12.py --part asr     # Parakeet-TDT throughput by batch size (nemo venv)
  python sim/capture_ch12.py --part tts     # Kokoro on the GPU and Pocket TTS on CPU processes, under load (tts venv)

Run from voice_applications/ on a GPU pod. Writes data/ch12/*.json.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
OUT = ROOT / "data" / "ch12"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LEVELS = [1, 2, 4, 8, 16, 32, 64]


def replay():
    """Every generation step of Chapter 8's main run, as the exact message list the model saw."""
    from receptionist import PROMPT_GUARDED
    log = json.loads((ROOT / "data" / "ch8" / "talk_4b-snap-guarded.json").read_text())["calls"]
    requests = []
    for cid, turns in log.items():
        history = []
        for t in turns:
            if not t["steps"]:
                continue
            history.append({"role": "user", "content": t["heard"]})
            for s in t["steps"]:
                requests.append({"call": cid, "messages": [{"role": "system", "content": PROMPT_GUARDED}] + json.loads(
                    json.dumps(history)), "was_tool_call": bool(s["calls"])})
                if s["calls"]:
                    ids = [f"call_{len(requests)}_{k}" for k in range(len(s["calls"]))]
                    history.append({"role": "assistant", "content": "", "tool_calls": [
                        {"id": i, "type": "function",
                         "function": {"name": c.get("name") or "invalid", "arguments": json.dumps(c.get("arguments", {}))}}
                        for i, c in zip(ids, s["calls"])]})
                    for i, c in zip(ids, s["calls"]):
                        history.append({"role": "tool", "tool_call_id": i, "content": json.dumps(c.get("result"))})
            history.append({"role": "assistant", "content": t["reply"]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replay.json").write_text(json.dumps(requests, indent=0, ensure_ascii=False))
    print(len(requests), "requests,", sum(r["was_tool_call"] for r in requests), "of them tool calls")


def llm():
    import asyncio
    import subprocess
    import urllib.request
    from openai import AsyncOpenAI
    from receptionist import TOOL_SCHEMAS
    server = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--port", "8000",
         "--max-model-len", "8192", "--gpu-memory-utilization", "0.85", "--enable-auto-tool-choice",
         "--tool-call-parser", "hermes"],
        stdout=open("/root/vllm_server.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(180):
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=2)
            break
        except Exception:
            time.sleep(5)
    else:
        server.kill()
        raise SystemExit("vLLM did not start; see /root/vllm_server.log")
    requests = json.loads((OUT / "replay.json").read_text())
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="local")

    async def one(req):
        t0 = time.perf_counter()
        first_token = first_sentence = None
        text, tokens = "", 0
        stream = await client.chat.completions.create(
            model=MODEL, messages=req["messages"], tools=TOOL_SCHEMAS, temperature=0, max_tokens=200, stream=True)
        async for chunk in stream:
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            piece = (d.content or "") + "".join(tc.function.arguments or "" for tc in (d.tool_calls or [])
                                                if tc.function)
            if piece:
                tokens += 1
                if first_token is None:
                    first_token = time.perf_counter() - t0
                text += d.content or ""
                if first_sentence is None and d.content and re.search(r"[.!?](\s|$)", text):
                    first_sentence = time.perf_counter() - t0
        return {"first_token": first_token, "first_sentence": first_sentence, "total": time.perf_counter() - t0,
                "chunks": tokens, "tool": req["was_tool_call"]}

    async def level(n, per_worker=None):
        per_worker = per_worker or max(4, 3 * len(requests) // n // 2)
        results = []

        async def worker(w):
            for k in range(per_worker):
                results.append(await one(requests[(w * 7 + k) % len(requests)]))

        t0 = time.perf_counter()
        await asyncio.gather(*(worker(w) for w in range(n)))
        return {"in_flight": n, "wall": time.perf_counter() - t0, "requests": results}

    async def main():
        await level(2, 4)  # warm up
        out = []
        for n in LEVELS:
            r = await level(n)
            ft = sorted(x["first_token"] for x in r["requests"] if x["first_token"])
            print(n, "in flight: median first token", round(ft[len(ft) // 2], 3), "s,",
                  len(r["requests"]), "requests in", round(r["wall"], 1), "s", flush=True)
            out.append(r)
        return out

    results = asyncio.run(main())
    server.terminate()
    (OUT / "llm_load.json").write_text(json.dumps({"model": MODEL, "server": "vLLM", "levels": results}, indent=0))


def asr():
    import nemo.collections.asr as nemo_asr
    import numpy as np
    import soundfile as sf
    from scipy import signal
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").cuda().eval()
    tmp = Path("/root/ch12")
    tmp.mkdir(exist_ok=True)
    paths = []
    for p in sorted((ROOT / "data" / "ch8" / "calls").glob("*.wav")):
        x, sr = sf.read(p, dtype="float32")
        sf.write(tmp / p.name, signal.resample_poly(x, 16000, sr).astype(np.float32), 16000)
        paths.append(str(tmp / p.name))
    audio_s = sum(sf.info(p).duration for p in paths)
    model.transcribe(paths[:8], batch_size=8, verbose=False)  # warm up
    rows = []
    for bs in [1, 4, 16, 64]:
        t0 = time.perf_counter()
        model.transcribe(paths, batch_size=bs, verbose=False)
        wall = time.perf_counter() - t0
        rows.append({"batch": bs, "clips": len(paths), "audio_s": audio_s, "wall": wall})
        print("batch", bs, round(wall, 2), "s for", len(paths), "clips", flush=True)
    (OUT / "asr_load.json").write_text(json.dumps(rows, indent=0))


def tts():
    import multiprocessing as mp
    import os
    import threading
    import numpy as np
    log = json.loads((ROOT / "data" / "ch8" / "talk_4b-snap-guarded.json").read_text())["calls"]
    firsts = sorted({re.split(r"(?<=[.!?])\s", t["reply"].strip())[0] for turns in log.values() for t in turns
                     if t["reply"].strip()})[:24]
    from kokoro import KModel, KPipeline
    kmodel = KModel().to("cuda").eval()
    pipes = [KPipeline(lang_code="b", model=kmodel) for _ in range(8)]
    list(pipes[0]("Warming up.", voice="bf_lily"))
    kokoro = []
    for n in [1, 4, 8]:
        times, lock = [], threading.Lock()

        def work(i):
            for s in firsts[i::n]:
                t0 = time.perf_counter()
                _, _, audio = next(iter(pipes[i](s, voice="bf_lily")))
                with lock:
                    times.append((time.perf_counter() - t0, len(audio) / 24000))

        t0 = time.perf_counter()
        threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        kokoro.append({"parallel": n, "wall": time.perf_counter() - t0, "requests": times})
        print("kokoro", n, round(float(np.median([a for a, _ in times])), 3), flush=True)
    pocket = []
    for n in [1, 2, 4, 8]:
        with mp.get_context("spawn").Pool(n) as pool:
            t0 = time.perf_counter()
            res = pool.map(_pocket_worker, [firsts[i::n] for i in range(n)])
            pocket.append({"processes": n, "threads_each": 2, "wall": time.perf_counter() - t0,
                           "requests": [x for r in res for x in r]})
        print("pocket", n, round(float(np.median([a for a, _, _ in pocket[-1]["requests"]])), 3), flush=True)
    (OUT / "tts_load.json").write_text(json.dumps({"cpus": os.cpu_count(), "kokoro_gpu": kokoro,
                                                   "pocket_cpu": pocket}, indent=0))


def _pocket_worker(sentences):
    import torch
    from pocket_tts import TTSModel
    torch.set_num_threads(2)
    model = TTSModel.load_model()
    state = model.get_state_for_audio_prompt("anna")
    model.generate_audio(state, "Warming up.")
    out = []
    for s in sentences:
        t0 = time.perf_counter()
        first = None
        n = 0
        for chunk in model.generate_audio_stream(state, s):
            first = first or time.perf_counter() - t0
            n += chunk.reshape(-1).shape[0]
        out.append((first, time.perf_counter() - t0, n / model.sample_rate))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["replay", "llm", "asr", "tts"])
    {"replay": replay, "llm": llm, "asr": asr, "tts": tts}[ap.parse_args().part]()
