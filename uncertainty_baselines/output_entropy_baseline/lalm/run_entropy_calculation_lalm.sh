#!/bin/bash

echo "======================================================"
echo "Output Entropy AUC Calculation for LALM Models"
echo "======================================================"
echo "Based on original calculate_entropy.py logic"
echo ""

cd "$(dirname "$0")"

# Clear the output file
output_file="entropy_auroc_results_lalm.txt"
> "$output_file"

echo "Output Entropy AUC Results for LALM Models" >> "$output_file"
echo "Generated on: $(date)" >> "$output_file"
echo "===========================================" >> "$output_file"
echo "" >> "$output_file"

# Define models, tasks, methods, and experiment types
models=("salmonn" "qwen" "desc_llm")
tasks=("count" "duration" "order")
methods=("sampling")
exp_types=("orig" "neg")

total_combinations=$((${#models[@]} * ${#tasks[@]} * ${#methods[@]} * ${#exp_types[@]}))
current=0

echo "Processing $total_combinations combinations..."
echo ""

for model in "${models[@]}"; do
    echo "========================================"
    echo "Processing model: $model"
    echo "========================================"
    
    for exp_type in "${exp_types[@]}"; do
        echo "  Experiment type: $exp_type"
        
        for task in "${tasks[@]}"; do
            echo "    Task: $task"
            
            for method in "${methods[@]}"; do
                current=$((current + 1))
                echo "      Method: $method [$current/$total_combinations]"
                
                # Run the entropy calculation
                result=$(python calculate_entropy_lalm_original.py \
                    --task "$task" \
                    --model "$model" \
                    --method "$method" \
                    --exp_type "$exp_type" 2>&1)
                
                # Extract AUC score from output
                auc=$(echo "$result" | grep "AUC-ROC:" | tail -1 | awk '{print $2}')
                
                if [[ -n "$auc" ]]; then
                    echo "        ✓ AUC-ROC: $auc"
                    echo "Model: $model, Task: $task, Method: $method, Exp: $exp_type, AUC-ROC: $auc" >> "$output_file"
                else
                    echo "        ✗ Error processing this combination"
                    echo "Model: $model, Task: $task, Method: $method, Exp: $exp_type, AUC-ROC: ERROR" >> "$output_file"
                    echo "Error details:" >> "$output_file"
                    echo "$result" >> "$output_file"
                    echo "---" >> "$output_file"
                fi
                
                echo ""
            done
        done
    done
    
    echo "Completed model: $model"
    echo ""
done

echo "======================================================"
echo "All calculations completed!"
echo ""
echo "Results summary:"
echo "======================================================"

# Display successful results
echo "Successful calculations:"
grep -v "ERROR" "$output_file" | grep "AUC-ROC:" | while read line; do
    echo "  $line"
done

echo ""
echo "Failed calculations:"
grep "ERROR" "$output_file" | while read line; do
    echo "  $line"
done

echo ""
echo "======================================================"
echo "Full results saved to: $output_file"
echo "Arrays saved to: ../output_sampling_arrays/"
echo "======================================================"
