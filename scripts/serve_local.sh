#!/usr/bin/env bash
# Start a llama.cpp server for one of the local GGUF models.
#
#   scripts/serve_local.sh qwen             # port 8080 (default rater)
#   scripts/serve_local.sh qwen-vl          # same model + vision projector (image input)
#   scripts/serve_local.sh gemma4-vl 8083   # Gemma 4 E4B + vision projector (a second model family)
#   scripts/serve_local.sh qwen27-vl 8084   # Qwen3.5-27B IQ4_XS + projector (~18 GB; stop other servers first)
#   scripts/serve_local.sh deepseek 8081    # second model, second port
#
# Wraps serve.sh from the inference_pipeline repo so the flags tuned for this
# machine (thread count, context size, q8_0 KV cache) stay in one place.
set -euo pipefail

PIPELINE="${INFERENCE_PIPELINE:-$HOME/personal/projects/inference_pipeline}"
NAME="${1:-qwen}"
PORT="${2:-8080}"

EXTRA=()
case "$NAME" in
  qwen)     MODEL="$PIPELINE/models/Qwen3.5-4B-Q4_K_M.gguf" ;;
  qwen-vl)  # Qwen3.5 is natively multimodal; the projector (unsloth
            # mmproj-F16.gguf, 672 MB) turns image patches into model tokens.
            # --image-max-tokens caps the tokens per receipt: CPU prefill is
            # ~14 tok/s, so every 1000 image tokens is over a minute.
            MODEL="$PIPELINE/models/Qwen3.5-4B-Q4_K_M.gguf"
            MMPROJ="$PIPELINE/models/mmproj-Qwen3.5-4B-F16.gguf"
            [[ -f "$MMPROJ" ]] || { echo "projector not found: $MMPROJ" >&2; exit 1; }
            # -b/-ub 2048: an image must fit in one micro-batch. At serve.sh's
            # -ub 512 a ~540-token receipt is split and the output is garbage.
            # --no-mmproj-offload: otherwise the vision encoder runs on the
            # Iris Xe via Vulkan even at -ngl 0; there it produced garbage and
            # then vk ErrorDeviceLost, which aborts the server.
            EXTRA=(--mmproj "$MMPROJ" --no-mmproj-offload --image-max-tokens "${IMAGE_MAX_TOKENS:-1024}"
                   -b 2048 -ub 2048) ;;
  gemma4-vl) # Google's Gemma 4 E4B: a different family from Qwen, so its
            # errors are less likely to be correlated with the Qwen rater's.
            # ~5.6 GB loaded; -c 8192 halves the KV cache vs serve.sh's 16k,
            # which is plenty for one receipt. Same image flags as qwen-vl.
            MODEL="$PIPELINE/models/gemma-4-E4B-it-Q4_0.gguf"
            # Q8_0 projector, not BF16: this CPU has no native BF16, and the
            # BF16 encoder took ~250 s per receipt image.
            MMPROJ="$PIPELINE/models/mmproj-gemma-4-E4B-it-Q8_0.gguf"
            [[ -f "$MMPROJ" ]] || { echo "projector not found: $MMPROJ" >&2; exit 1; }
            EXTRA=(--mmproj "$MMPROJ" --no-mmproj-offload --image-max-tokens "${IMAGE_MAX_TOKENS:-1024}"
                   -b 2048 -ub 2048 -c 8192) ;;
  qwen27-vl) # Qwen3.5-27B at IQ4_XS (15 GB), the largest quant that leaves
            # headroom on a 30 GB laptop. ~1 tok/s decode and ~2 tok/s prefill
            # on this CPU: about 10 min per receipt. -c 8192 keeps the KV small.
            MODEL="$PIPELINE/models/Qwen3.5-27B-IQ4_XS.gguf"
            MMPROJ="$PIPELINE/models/mmproj-Qwen3.5-27B-F16.gguf"
            [[ -f "$MMPROJ" ]] || { echo "projector not found: $MMPROJ" >&2; exit 1; }
            EXTRA=(--mmproj "$MMPROJ" --no-mmproj-offload --image-max-tokens "${IMAGE_MAX_TOKENS:-1024}"
                   -b 2048 -ub 2048 -c 8192) ;;
  deepseek) MODEL="$PIPELINE/models/DeepSeek-Coder-V2-Lite-IQ4_XS.gguf" ;;
  *)        MODEL="$NAME" ;;   # or pass an explicit .gguf path
esac

[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }
echo "serving $(basename "$MODEL") on port $PORT"

# VULKAN=1 nearly triples prefill on the iGPU and slows decode. Extraction is
# prefill-heavy (a receipt in, a few dozen JSON tokens out), so it is worth
# measuring both ways on your own documents.
MODEL="$MODEL" PORT="$PORT" exec "$PIPELINE/serve.sh" "${EXTRA[@]}"
