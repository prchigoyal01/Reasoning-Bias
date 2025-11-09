#!/bin/bash

# MBBQ Evaluation Sweep Script with Memory Management
set +e  # Don't exit on error

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR="src/results"
DATA_DIR="MBBQ_data"
MAIN_SCRIPT="src/main.py"
BATCH_SIZE=2  # Reduced batch size

# Models
LLAMA_3_1="meta-llama/Llama-3.1-8B-Instruct"
LLAMA_2="meta-llama/Llama-2-7b-chat-hf"
DEEPSEEK_R1="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Languages
LANGUAGES=("en" "es" "tr" "nl")

# Categories
CATEGORIES=(
    "Age"
    "Age_control"
    "Disability_status"
    "Disability_status_control"
    "Gender_identity"
    "Gender_identity_control"
    "Physical_appearance"
    "Physical_appearance_control"
    "SES"
    "SES_control"
    "Sexual_orientation"
    "Sexual_orientation_control"
)

# ============================================================================
# Helper Functions
# ============================================================================

run_experiment() {
    local model=$1
    local eval_mode=$2
    local language=$3
    local category=$4
    
    echo ""
    echo "========================================================================"
    echo "Running: $eval_mode | $language | $category"
    echo "Model: $model"
    echo "========================================================================"
    
    python $MAIN_SCRIPT \
        --model_name "$model" \
        --eval_mode "$eval_mode" \
        --language "$language" \
        --category "$category" \
        --batch_size $BATCH_SIZE \
        --output_dir "$OUTPUT_DIR" \
        --data_dir "$DATA_DIR" \
    
    local exit_code=$?
    
    # Clear GPU memory after each run
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5  # Wait for memory to clear
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ SUCCESS"
        return 0
    else
        echo "✗ FAILED (exit code: $exit_code)"
        return 1
    fi
}

# ============================================================================
# Main Sweep
# ============================================================================

mkdir -p "$OUTPUT_DIR"

SUCCESS_COUNT=0
FAILURE_COUNT=0
FAILED_RUNS_FILE="$OUTPUT_DIR/failed_runs.txt"
> "$FAILED_RUNS_FILE"  # Clear file

echo "========================================================================"
echo "MBBQ EVALUATION SWEEP"
echo "========================================================================"

# ============================================================================
# Process each model separately to manage memory
# ============================================================================


# SHORT ANSWER - Llama 2
# echo ""
# echo "=== SHORT ANSWER: Llama 2 ==="
# for language in "${LANGUAGES[@]}"; do
#     for category in "${CATEGORIES[@]}"; do
#         if run_experiment "$LLAMA_2" "short_answer" "$language" "$category"; then
#             ((SUCCESS_COUNT++))
#         else
#             ((FAILURE_COUNT++))
#             echo "short_answer|$LLAMA_2|$language|$category" >> "$FAILED_RUNS_FILE"
#         fi
#     done
# done

# COT - Llama 3.1
echo ""
echo "=== COT: Llama 3.1 ==="
for language in "${LANGUAGES[@]}"; do
    for category in "${CATEGORIES[@]}"; do
        if run_experiment "$LLAMA_3_1" "cot" "$language" "$category"; then
            ((SUCCESS_COUNT++))
        else
            ((FAILURE_COUNT++))
            echo "cot|$LLAMA_3_1|$language|$category" >> "$FAILED_RUNS_FILE"
        fi
    done
done

# COT - Llama 2
# echo ""
# echo "=== COT: Llama 2 ==="
# for language in "${LANGUAGES[@]}"; do
#     for category in "${CATEGORIES[@]}"; do
#         if run_experiment "$LLAMA_2" "cot" "$language" "$category"; then
#             ((SUCCESS_COUNT++))
#         else
#             ((FAILURE_COUNT++))
#             echo "cot|$LLAMA_2|$language|$category" >> "$FAILED_RUNS_FILE"
#         fi
#     done
# done

# REASONING - DeepSeek-R1
echo ""
echo "=== REASONING: DeepSeek-R1 ==="
for language in "${LANGUAGES[@]}"; do
    for category in "${CATEGORIES[@]}"; do
        if run_experiment "$DEEPSEEK_R1" "reasoning" "$language" "$category"; then
            ((SUCCESS_COUNT++))
        else
            ((FAILURE_COUNT++))
            echo "reasoning|$DEEPSEEK_R1|$language|$category" >> "$FAILED_RUNS_FILE"
        fi
    done
done

# ============================================================================
# Summary
# ============================================================================

TOTAL_RUNS=$((SUCCESS_COUNT + FAILURE_COUNT))

echo ""
echo "========================================================================"
echo "SWEEP SUMMARY"
echo "========================================================================"
echo "Total Runs:    $TOTAL_RUNS"
echo "✓ Successful:  $SUCCESS_COUNT"
echo "✗ Failed:      $FAILURE_COUNT"
if [ $TOTAL_RUNS -gt 0 ]; then
    echo "Success Rate:  $(awk "BEGIN {printf \"%.1f\", ($SUCCESS_COUNT/$TOTAL_RUNS)*100}")%"
fi
echo "========================================================================"

if [ $FAILURE_COUNT -gt 0 ]; then
    echo ""
    echo "Failed runs saved to: $FAILED_RUNS_FILE"
    cat "$FAILED_RUNS_FILE"
fi

echo ""
echo "✓ Sweep complete!"