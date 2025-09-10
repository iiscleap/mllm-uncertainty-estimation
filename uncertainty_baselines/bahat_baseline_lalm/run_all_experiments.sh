#!/bin/bash

DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "Performing a dry run - no commands will be executed."
fi

echo "Running comprehensive uncertainty evaluation for all models and tasks"
echo "===================================================================="

MODELS=("desc_llm" "qwen" "salmonn")
TASKS=("count" "order" "duration")

# Create summary results file
SUMMARY_FILE="summary_all_results.txt"
echo "Comprehensive Uncertainty Evaluation Results" > $SUMMARY_FILE
echo "Generated on: $(date)" >> $SUMMARY_FILE
echo "=============================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

total_experiments=$((${#MODELS[@]} * ${#TASKS[@]}))
current_experiment=0

for MODEL in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        current_experiment=$((current_experiment + 1))
        echo ""
        echo "[$current_experiment/$total_experiments] Running $MODEL - $TASK"
        echo "================================================="
        
        # Run single model evaluation
        if [ "$DRY_RUN" = true ]; then
            echo "Dry run: ./run_single_model.sh $MODEL $TASK"
        else
            ./run_single_model.sh $MODEL $TASK
        fi
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed $MODEL - $TASK"
            
            # Add results to summary
            echo "$MODEL - $TASK:" >> $SUMMARY_FILE
            if [ -f "entropy_results_${MODEL}_${TASK}.txt" ]; then
                cat "entropy_results_${MODEL}_${TASK}.txt" >> $SUMMARY_FILE
            else
                echo "  Results file not found" >> $SUMMARY_FILE
            fi
            echo "" >> $SUMMARY_FILE
            
        else
            echo "✗ Failed to complete $MODEL - $TASK"
            echo "$MODEL - $TASK: FAILED" >> $SUMMARY_FILE
            echo "" >> $SUMMARY_FILE
        fi
        
        echo "================================================="
    done
done

echo ""
echo "===================================================================="
echo "All experiments completed!"
echo "Summary results saved in: $SUMMARY_FILE"
echo ""

cat $SUMMARY_FILE
