#!/bin/bash
# Run every capture step on two items, on this machine's GPU, to check it fits and works before a full rerun.
#   VOTF_FP16=1 bash tools/smoke_test.sh        (from the book folder, voice/ in the companion repo)
# VOTF_FP16=1 takes the float16 path a Colab T4 would. Writes into the data/ folder, so run it on a copy.
# Steps known not to fit 16 GB in float16: Fish S2-Pro, Qwen3-8B, Moshi (see the README).
set -u
export VOTF_QUICK=${VOTF_QUICK:-2}
LOG=${LOG:-/tmp/votf-smoke}
mkdir -p "$LOG"
job() {  # job <name> <stack> <script and arguments...>
  local name=$1 stack=$2; shift 2
  bash tools/stack.sh "$stack" > "$LOG/stack_$stack.log" 2>&1
  if "${VOTF_ENVS:-$HOME/.votf/envs}/$stack/bin/python" "$@" > "$LOG/$name.log" 2>&1; then echo "PASS $name"
  else echo "FAIL $name: $(grep -E 'Error|error:' "$LOG/$name.log" | tail -n 1 | cut -c1-160)"; fi
}
job calls tts sim/halfwheel_calls.py
job librispeech nemo sim/librispeech_subset.py
job ch2 nemo sim/capture_ch2.py
job ch3 nemo sim/capture_ch3.py
job ch4 nemo sim/capture_ch4.py
job ch5-turns tts sim/turn_calls.py
job ch5-nemo nemo sim/capture_ch5.py --part nemo
job ch5-kyutai moshi sim/capture_ch5.py --part kyutai
job ch5-synth tts sim/aec_ch5.py --synth
job ch5-apply nemo sim/aec_ch5.py --apply
for p in voice:nemo kokoro:tts pocket:tts kyutai:moshi csm:hf asr:nemo; do job ch6-${p%%:*} ${p##*:} sim/capture_ch6.py --part ${p%%:*}; done
for p in kokoro:tts kyutai:moshi fish1:fish1 fish2:fish2 refs:nemo asr:nemo judge:hf; do job ch7-${p%%:*} ${p##*:} sim/capture_ch7.py --part ${p%%:*}; done
job ch8-voice tts sim/capture_ch8.py --part voice
job ch8-hear nemo sim/capture_ch8.py --part hear
job ch8-talk-4b hf sim/capture_ch8.py --part talk --only 4b-snap-guarded 4b-snap-code
job ch8-speak tts sim/capture_ch8.py --part speak
for p in fetch:nemo voice:tts hear:nemo judge:hf; do job ch9-${p%%:*} ${p##*:} sim/capture_ch9.py --part ${p%%:*}; done
for p in moshi:moshi refs:nemo csm:hf; do job ch10-${p%%:*} ${p##*:} sim/capture_ch10.py --part ${p%%:*}; done
for p in voice:tts hear:nemo talk:hf judge:hf; do job ch11-${p%%:*} ${p##*:} sim/capture_ch11.py --part ${p%%:*}; done
for p in llm:vllm asr:nemo tts:tts; do job ch12-${p%%:*} ${p##*:} sim/capture_ch12.py --part ${p%%:*}; done
