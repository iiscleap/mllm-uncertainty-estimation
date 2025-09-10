#!/bin/bash

# Usage: ./run_single_model.sh <model> <task>
# model: desc_llm, qwen, salmonn
# task: count, order, duration

# if [ "$#" -ne 2 ]; then
#     echo "Usage: $0 <model> <task>"
#     echo "model: desc_llm, qwen, salmonn"
#     echo "task: count, order, duration"
#     exit 1
# fi

MODEL=$1

echo "Running complete evaluation for $MODEL on all tasks"
echo "============================================="

echo "Setting up environment..."
if [ "$MODEL" == "salmonn" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate salmonn
fi


# Create results directory if it doesn't exist
mkdir -p "${MODEL}_results"

# Clear previous results for this task
echo "Clearing previous results for $MODEL..."
rm -f "entropy_results_${MODEL}.txt"

echo "Experiment types: text_audio"
echo ""

# Run model inference for all three experiment types
for TASK in "order" "duration"; do
    echo "Running $MODEL inference for $TASK - text_audio..."
    
    case $MODEL in
        desc_llm)
            python desc_llm_code_modified.py $TASK text_audio
            ;;
        qwen)
            python qwen_audio_modified.py $TASK text_audio
            ;;
        salmonn)
            python salmonn_modified.py $TASK text_audio
            ;;
        *)
            echo "Unknown model: $MODEL"
            exit 1
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed inference for $MODEL - $TASK - text_audio"
    else
        echo "✗ Failed inference for $MODEL - $TASK - text_audio"
        exit 1
    fi
    echo ""
done

# Calculate entropy and AUC for all experiment types
# echo "Calculating entropy and AUC scores..."
# for EXP_TYPE in "text_audio"; do
#     echo "Processing $EXP_TYPE..."
#     python calculate_entropy_comprehensive.py $MODEL $TASK $EXP_TYPE
#     if [ $? -eq 0 ]; then
#         echo "✓ Completed entropy calculation for $EXP_TYPE"
#     else
#         echo "✗ Failed entropy calculation for $EXP_TYPE"
#     fi
# done

# echo ""
# echo "============================================="
# echo "Results saved in: entropy_results_${MODEL}_${TASK}.txt"
# echo "Model outputs saved in: ${MODEL}_results/"

# # Display final results
# if [ -f "entropy_results_${MODEL}_${TASK}.txt" ]; then
#     echo ""
#     echo "Final AUC Results for $MODEL - $TASK:"
#     cat "entropy_results_${MODEL}_${TASK}.txt"
# fi

echo "Evaluation complete for $MODEL"
