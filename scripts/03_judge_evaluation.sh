#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_NAME="$REPO_ROOT/src/03_judge_evaluation.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY is not set. Export it before running this script." >&2
    exit 1
fi

export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"

cd "$REPO_ROOT"

# Configuration format: INPUT_FILE|OUTPUT_FILE|MODEL|BATCH_SIZE
# INPUT_FILE: evaluation output from 02_evaluate_model.py
# OUTPUT_FILE: judge output filename
# MODEL: judge model (e.g., openai/gpt-5-mini)
# BATCH_SIZE: concurrent API calls

CONFIGS=(
    "results/evaluation_mistral-ocr_direct_without_policy_gpt-5-2_thinking.json|judge_mistral-ocr_direct_without_policy_gpt-5-2_thinking.json|openai/gpt-5-mini|16"
    "results/evaluation_mistral-ocr_indirect_without_policy_gpt-5-2_thinking.json|judge_mistral-ocr_indirect_without_policy_gpt-5-2_thinking.json|openai/gpt-5-mini|16"
    "results/evaluation_mistral-ocr_direct_without_policy_gemini3_pro.json|judge_mistral-ocr_direct_without_policy_gemini3_pro.json|openai/gpt-5-mini|16"
    "results/evaluation_mistral-ocr_indirect_without_policy_gemini3_pro.json|judge_mistral-ocr_indirect_without_policy_gemini3_pro.json|openai/gpt-5-mini|16"
    "results/evaluation_mistral-ocr_direct_with_policy_gemini3_pro.json|judge_mistral-ocr_direct_with_policy_gemini3_pro.json|openai/gpt-5-mini|16"
    "results/evaluation_mistral-ocr_indirect_with_policy_gemini3_pro.json|judge_mistral-ocr_indirect_with_policy_gemini3_pro.json|openai/gpt-5-mini|16"
    "results/evaluation_mistral-ocr_indirect_with_policy_qwen3-vl-235b_thinking.json|judge_mistral-ocr_indirect_with_policy_qwen3-vl-235b_thinking.json|openai/gpt-5-mini|16"
    "results/evaluation_mistral-ocr_direct_with_policy_qwen3-vl-235b_thinking.json|judge_mistral-ocr_direct_with_policy_qwen3-vl-235b_thinking.json|openai/gpt-5-mini|16"
    "results/evaluation_mistral-ocr_indirect_without_policy_qwen3-vl-235b_thinking.json|judge_mistral-ocr_indirect_without_policy_qwen3-vl-235b_thinking.json|openai/gpt-5-mini|16"
    "results/evaluation_mistral-ocr_direct_without_policy_qwen3-vl-235b_thinking.json|judge_mistral-ocr_direct_without_policy_qwen3-vl-235b_thinking.json|openai/gpt-5-mini|16"
)

# Run each configuration
for config in "${CONFIGS[@]}"; do
    IFS='|' read -r INPUT_FILE OUTPUT_FILE MODEL BATCH_SIZE <<< "$config"

    # Convert - to empty
    [ "$MODEL" = "-" ] && MODEL=""
    [ "$BATCH_SIZE" = "-" ] && BATCH_SIZE=""

    ARGS="--input-file $INPUT_FILE --output-file $OUTPUT_FILE"

    [ -n "$MODEL" ] && ARGS="$ARGS --model $MODEL"
    [ -n "$BATCH_SIZE" ] && ARGS="$ARGS --batch-size $BATCH_SIZE"

    echo "========================================"
    echo "Running: $PYTHON_BIN $SCRIPT_NAME $ARGS"
    echo "========================================"
    "$PYTHON_BIN" "$SCRIPT_NAME" $ARGS
done
