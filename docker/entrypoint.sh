#!/bin/bash
# NInfer V100 entrypoint: maps environment variables to ninfer-serve flags.
# All defaults mirror the production recipe documented in docs/V100-BUILD.md.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/qwen3_8_27b_nvfp4.ninfer}"
MODEL_ID="${MODEL_ID:-qwen3.8-27b}"
PORT="${PORT:-7106}"
DEVICE="${DEVICE:-0}"
MAX_CONTEXT="${MAX_CONTEXT:-230000}"
PREFILL_CHUNK="${PREFILL_CHUNK:-2048}"
KV_DTYPE="${KV_DTYPE:-int8}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-2}"
HOST_STATE_SLOTS="${HOST_STATE_SLOTS:-8}"
HOST_KV_MIB="${HOST_KV_MIB:-8192}"
DRAFT_TOKENS="${DRAFT_TOKENS:-3}"
SPEC="${SPEC:-mtp}"
PENDING_TIMEOUT_MS="${PENDING_TIMEOUT_MS:-600000}"
VISION="${VISION:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Optional one-shot model download (~23.7 GB). Set AUTO_DOWNLOAD=1 to enable.
# Point HF_ENDPOINT at a mirror (e.g. https://hf-mirror.com) if huggingface.co
# is unreachable from your network.
if [ ! -f "$MODEL_PATH" ] && [ "${AUTO_DOWNLOAD:-0}" = "1" ]; then
    HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
    HF_REPO="${HF_REPO:-neroued/Qwen3.8-27B-nvfp4-NInfer}"
    HF_FILE="${HF_FILE:-qwen3_8_27b_nvfp4.ninfer}"
    mkdir -p "$(dirname "$MODEL_PATH")"
    echo "Downloading ${HF_REPO}/${HF_FILE} from ${HF_ENDPOINT} (~23.7 GB) ..."
    curl -fL --retry 5 -C - -o "$MODEL_PATH" \
        "${HF_ENDPOINT}/${HF_REPO}/resolve/main/${HF_FILE}"
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: model artifact not found at $MODEL_PATH" >&2
    echo "Mount the model directory into the container and set MODEL_PATH," >&2
    echo "or set AUTO_DOWNLOAD=1 (plus HF_ENDPOINT for mirrors) to fetch it." >&2
    exit 1
fi

ARGS=(
    "$MODEL_PATH"
    --host 0.0.0.0
    --port "$PORT"
    --device "$DEVICE"
    --model-id "$MODEL_ID"
    --max-context "$MAX_CONTEXT"
    --prefill-chunk "$PREFILL_CHUNK"
    --kv-capacity auto
    --kv-dtype "$KV_DTYPE"
    --max-concurrency "$MAX_CONCURRENCY"
    --device-state-slots "$MAX_CONCURRENCY"
    --host-state-slots "$HOST_STATE_SLOTS"
    --host-kv-mib "$HOST_KV_MIB"
    --preserve-thinking
    --pending-timeout-ms "$PENDING_TIMEOUT_MS"
    --log-level "$LOG_LEVEL"
)

if [ "$SPEC" != "none" ]; then
    ARGS+=(--spec "$SPEC" --draft-tokens "$DRAFT_TOKENS" --lm-head-draft)
fi

if [ "$VISION" = "1" ]; then
    ARGS+=(--vision)
fi

if [ -n "${API_KEY:-}" ]; then
    ARGS+=(--api-key "$API_KEY")
fi

echo "Starting: ninfer-serve ${ARGS[*]}" >&2
exec /opt/ninfer/ninfer-serve "${ARGS[@]}"
