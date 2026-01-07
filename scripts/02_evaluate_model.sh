#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_NAME="$REPO_ROOT/src/02_evaluate_model.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY is not set. Export it before running this script." >&2
    exit 1
fi

export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"

cd "$REPO_ROOT"

# Define the configuration set (one entry per line).
# Format: INPUT_FILE|OUTPUT_FILE|QUERY_TYPE|WITH_POLICY|DOCS_DIR|OUTPUT_DIR|MODEL|PDF_ENGINE|DOC_MODE|MAX_PAGES|IMAGE_RESOLUTION|BATCH_SIZE|ENABLE_THINKING
# Use '-' to represent an empty value.
# WITH_POLICY is true/false (true adds the --with-policy flag).
# ENABLE_THINKING is true/false (true adds the --enable-thinking flag).

# CONFIGS=(
#     "data/03_final_data.json|evaluation_direct_with_policy.json|direct|true|-|-|-|-|8|false"
#     "data/03_final_data.json|evaluation_direct_without_policy.json|direct|false|-|-|-|-|8|false"
#     "data/03_final_data.json|evaluation_indirect_with_policy.json|indirect|true|-|-|-|-|pdf|-|-|8|false"
#     "data/03_final_data.json|evaluation_indirect_without_policy.json|indirect|false|-|-|-|-|pdf|-|-|8|false"
# )
CONFIGS=(
    "data/03_final_data.json|evaluation_image_indirect_without_policy_mistral-large_instruct.json|indirect|false|-|-|mistralai/mistral-large-2512|-|image|120|144|8|false"
    "data/03_final_data.json|evaluation_image_direct_without_policy_mistral-large_instruct.json|direct|false|-|-|mistralai/mistral-large-2512|-|image|120|144|8|false"
)

# Run each configuration
for config in "${CONFIGS[@]}"; do
    IFS='|' read -r INPUT_FILE OUTPUT_FILE QUERY_TYPE WITH_POLICY DOCS_DIR OUTPUT_DIR MODEL PDF_ENGINE DOC_MODE MAX_PAGES IMAGE_RESOLUTION BATCH_SIZE ENABLE_THINKING <<< "$config"

    # Treat '-' as an empty value
    [ "$DOCS_DIR" = "-" ] && DOCS_DIR=""
    [ "$OUTPUT_DIR" = "-" ] && OUTPUT_DIR=""
    [ "$MODEL" = "-" ] && MODEL=""
    [ "$PDF_ENGINE" = "-" ] && PDF_ENGINE=""
    [ "$DOC_MODE" = "-" ] && DOC_MODE=""
    [ "$MAX_PAGES" = "-" ] && MAX_PAGES=""
    [ "$IMAGE_RESOLUTION" = "-" ] && IMAGE_RESOLUTION=""
    [ "$BATCH_SIZE" = "-" ] && BATCH_SIZE=""

    ARGS="--input-file $INPUT_FILE --output-file $OUTPUT_FILE --query-type $QUERY_TYPE"

    # Append --with-policy when requested
    [ "$WITH_POLICY" = "true" ] && ARGS="$ARGS --with-policy"

    # Append --enable-thinking when requested
    [ "$ENABLE_THINKING" = "true" ] && ARGS="$ARGS --enable-thinking"

    [ -n "$DOCS_DIR" ] && ARGS="$ARGS --docs-dir $DOCS_DIR"
    [ -n "$OUTPUT_DIR" ] && ARGS="$ARGS --output-dir $OUTPUT_DIR"
    [ -n "$MODEL" ] && ARGS="$ARGS --model $MODEL"
    [ -n "$PDF_ENGINE" ] && ARGS="$ARGS --pdf-engine $PDF_ENGINE"
    [ -n "$DOC_MODE" ] && ARGS="$ARGS --doc-mode $DOC_MODE"
    [ -n "$MAX_PAGES" ] && ARGS="$ARGS --max-pages $MAX_PAGES"
    [ -n "$IMAGE_RESOLUTION" ] && ARGS="$ARGS --image-resolution $IMAGE_RESOLUTION"
    [ -n "$BATCH_SIZE" ] && ARGS="$ARGS --batch-size $BATCH_SIZE"

    echo "Running: $PYTHON_BIN $SCRIPT_NAME $ARGS"
    "$PYTHON_BIN" "$SCRIPT_NAME" $ARGS
done
