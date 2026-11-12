#!/bin/bash
# Create one of the book's software stacks as its own virtual environment (in Colab or on any Linux GPU box).
#   bash tools/stack.sh nemo|tts|moshi|hf|vllm|fish1|fish2   ->  $VOTF_ENVS/<stack>/bin/python
# Each capture step runs in the stack it needs, so stacks with conflicting pins never share an environment.
set -e
stack=$1
SUDO=$([ "$(id -u)" = 0 ] || echo sudo)  # Colab runs as root; elsewhere apt needs sudo
envs=${VOTF_ENVS:-$HOME/.votf/envs}
command -v uv >/dev/null || pip install -q uv
if [ -f "$envs/$stack/.ok" ]; then echo "$stack ready"; exit 0; fi  # written only after a complete install
uv venv -q --python 3.12 "$envs/$stack"
P="uv pip install -q --python $envs/$stack/bin/python"
TORCH="torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124"
case $stack in
  nemo) $P $TORCH; $P torch==2.6.0 "nemo_toolkit[asr]==3.0.0" qwen-asr silero-vad pyarrow soundfile scipy ;;
  tts) (command -v espeak-ng >/dev/null || ($SUDO apt-get update -qq && $SUDO apt-get install -y -qq espeak-ng >/dev/null))
       $P $TORCH; $P torch==2.6.0 "kokoro>=0.9.4" pocket-tts soundfile scipy "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" ;;
  moshi) $P $TORCH; $P torch==2.6.0 "moshi==0.2.13" soundfile scipy ;;
  hf) $P $TORCH; $P torch==2.6.0 "transformers==5.22.2" "accelerate==1.15.0" librosa soundfile scipy numpy ;;
  vllm) $P vllm openai ninja ;;
  fish1) $P $TORCH; $P torch==2.6.0 transformers lightning hydra-core natsort einops librosa rich loguru loralib pyrootutils descript-audiotools "vector_quantize_pytorch==1.14.24" resampy "einx[torch]==0.2.2" opencc ormsgpack soundfile cachetools tiktoken numpy huggingface_hub pydantic "git+https://github.com/descriptinc/descript-audio-codec" ;;
  fish2) $P torch==2.8.0 torchaudio==2.8.0 "transformers==4.57.3" lightning hydra-core natsort einops librosa rich loguru loralib pyrootutils resampy "einx[torch]==0.2.2" zstandard ormsgpack "tiktoken>=0.8.0" "pydantic==2.9.2" cachetools descript-audio-codec descript-audiotools safetensors opencc-python-reimplemented soundfile click --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 ;;
  *) echo "unknown stack $stack"; exit 1 ;;
esac
"$envs/$stack/bin/python" -c "import torch; print('$stack ready, torch', torch.__version__, 'GPU', torch.cuda.is_available())"
touch "$envs/$stack/.ok"
