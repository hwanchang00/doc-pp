#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DVA_SCRIPT="$REPO_ROOT/src/04_mitigation_dva_provider_fixed.py"
JUDGE_SCRIPT="$REPO_ROOT/src/04_mitigation_dva_judge.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY is not set. Export it before running this script." >&2
    exit 1
fi

export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"

cd "$REPO_ROOT"

# ============================================
# Stage 1: DVA Evaluation
# ============================================
# Format: INPUT_FILE|OUTPUT_FILE|QUERY_TYPE|DOCS_DIR|OUTPUT_DIR|MODEL|PDF_ENGINE|DOC_MODE|MAX_PAGES|IMAGE_RESOLUTION|BATCH_SIZE|ENABLE_THINKING

DVA_CONFIGS=(
    "data/03_final_data.json|mitigation_dva_image_indirect_with_policy_qwen3-vl-235b_instruct.json|indirect|-|-|qwen/qwen3-vl-235b-a22b-instruct|-|image|120|144|4|false"
    "data/03_final_data.json|mitigation_dva_image_indirect_with_policy_mistral-large_instruct.json|indirect|-|-|mistralai/mistral-large-2512|-|image|120|144|4|false"
)

echo "============================================"
echo "Stage 1: Running DVA Evaluation"
echo "============================================"

for config in "${DVA_CONFIGS[@]}"; do
    IFS='|' read -r INPUT_FILE OUTPUT_FILE QUERY_TYPE DOCS_DIR OUTPUT_DIR MODEL PDF_ENGINE DOC_MODE MAX_PAGES IMAGE_RESOLUTION BATCH_SIZE ENABLE_THINKING <<< "$config"

    [ "$DOCS_DIR" = "-" ] && DOCS_DIR=""
    [ "$OUTPUT_DIR" = "-" ] && OUTPUT_DIR=""
    [ "$MODEL" = "-" ] && MODEL=""
    [ "$PDF_ENGINE" = "-" ] && PDF_ENGINE=""
    [ "$DOC_MODE" = "-" ] && DOC_MODE=""
    [ "$MAX_PAGES" = "-" ] && MAX_PAGES=""
    [ "$IMAGE_RESOLUTION" = "-" ] && IMAGE_RESOLUTION=""
    [ "$BATCH_SIZE" = "-" ] && BATCH_SIZE=""

    ARGS="--input-file $INPUT_FILE --output-file $OUTPUT_FILE --query-type $QUERY_TYPE"

    [ "$ENABLE_THINKING" = "true" ] && ARGS="$ARGS --enable-thinking"

    [ -n "$DOCS_DIR" ] && ARGS="$ARGS --docs-dir $DOCS_DIR"
    [ -n "$OUTPUT_DIR" ] && ARGS="$ARGS --output-dir $OUTPUT_DIR"
    [ -n "$MODEL" ] && ARGS="$ARGS --model $MODEL"
    [ -n "$PDF_ENGINE" ] && ARGS="$ARGS --pdf-engine $PDF_ENGINE"
    [ -n "$DOC_MODE" ] && ARGS="$ARGS --doc-mode $DOC_MODE"
    [ -n "$MAX_PAGES" ] && ARGS="$ARGS --max-pages $MAX_PAGES"
    [ -n "$IMAGE_RESOLUTION" ] && ARGS="$ARGS --image-resolution $IMAGE_RESOLUTION"
    [ -n "$BATCH_SIZE" ] && ARGS="$ARGS --batch-size $BATCH_SIZE"

    echo "Running: $PYTHON_BIN $DVA_SCRIPT $ARGS"
    "$PYTHON_BIN" "$DVA_SCRIPT" $ARGS
done

# ============================================
# Stage 2: Judge Evaluation (using final_answer)
# ============================================
# Format: INPUT_FILE|OUTPUT_FILE|JUDGE_MODEL|BATCH_SIZE

JUDGE_CONFIGS=(
    "results/mitigation_dva_image_indirect_with_policy_qwen3-vl-235b_instruct.json|judge_mitigation_dva_image_indirect_with_policy_qwen3-vl-235b_instruct.json|openai/gpt-5-mini|4"
    "results/mitigation_dva_image_indirect_with_policy_mistral-large_instruct.json|judge_mitigation_dva_image_indirect_with_policy_mistral-large_instruct.json|openai/gpt-5-mini|4"
)

echo ""
echo "============================================"
echo "Stage 2: Running Judge Evaluation"
echo "============================================"

for config in "${JUDGE_CONFIGS[@]}"; do
    IFS='|' read -r INPUT_FILE OUTPUT_FILE JUDGE_MODEL BATCH_SIZE <<< "$config"

    ARGS="--input-file $INPUT_FILE --output-file $OUTPUT_FILE --model $JUDGE_MODEL --batch-size $BATCH_SIZE"

    echo "Running: $PYTHON_BIN $JUDGE_SCRIPT $ARGS"
    "$PYTHON_BIN" "$JUDGE_SCRIPT" $ARGS
done

echo ""
echo "============================================"
echo "All stages completed!"
echo "============================================"
