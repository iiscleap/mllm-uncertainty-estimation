#!/bin/bash

echo "Running comprehensive VLM uncertainty evaluation for all models and datasets"
echo "========================================================================="

MODELS=("llava" "gemma3" "qwenvl" "pixtral" "phi4")
DATASETS=("blink" "vsr")

# Create summary results file
SUMMARY_FILE="summary_all_results_vlm.txt"
echo "Comprehensive VLM Uncertainty Evaluation Results" > $SUMMARY_FILE
echo "Generated on: $(date)" >> $SUMMARY_FILE
echo "=============================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

total_experiments=$((${#MODELS[@]} * ${#DATASETS[@]}))
current_experiment=0

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        current_experiment=$((current_experiment + 1))
        echo ""
        echo "[$current_experiment/$total_experiments] Running $MODEL - $DATASET"
        echo "================================================="
        
        # Run single model evaluation
        ./run_single_model_vlm.sh $MODEL $DATASET
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed $MODEL - $DATASET"
            
            # Add results to summary
            echo "$MODEL - $DATASET:" >> $SUMMARY_FILE
            if [ -f "entropy_results_vlm_${MODEL}_${DATASET}.txt" ]; then
                cat "entropy_results_vlm_${MODEL}_${DATASET}.txt" >> $SUMMARY_FILE
            else
                echo "  Results file not found" >> $SUMMARY_FILE
            fi
            echo "" >> $SUMMARY_FILE
            
        else
            echo "✗ Failed to complete $MODEL - $DATASET"
            echo "$MODEL - $DATASET: FAILED" >> $SUMMARY_FILE
            echo "" >> $SUMMARY_FILE
        fi
        
        echo "================================================="
    done
done

echo ""
echo "========================================================================="
echo "All VLM experiments completed!"
echo "Summary results saved in: $SUMMARY_FILE"
echo ""
echo "Individual results files:"
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        if [ -f "entropy_results_vlm_${MODEL}_${DATASET}.txt" ]; then
            echo "  - entropy_results_vlm_${MODEL}_${DATASET}.txt"
        fi
    done
done

echo ""
echo "Model output directories:"
for MODEL in "${MODELS[@]}"; do
    echo "  - ${MODEL}_results/"
done

echo ""
echo "Final Summary:"
echo "=============="
cat $SUMMARY_FILE
