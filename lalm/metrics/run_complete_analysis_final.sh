#!/bin/bash

# Complete analysis script for LALM experiments
# This script runs comprehensive metrics, generates all plots, and saves results as CSV
# All results will be saved in /home/debarpanb/VLM_project/malay_works/FESTA-uncertainty-estimation/results/lalm

RESULTS_DIR="/home/debarpanb/VLM_project/malay_works/FESTA-uncertainty-estimation/results/lalm"

echo "======================================================================="
echo "COMPLETE ANALYSIS FOR LALM UNCERTAINTY ESTIMATION"
echo "======================================================================="
echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Create results directory structure
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/risk_coverage_plots"
mkdir -p "$RESULTS_DIR/ece_calibration_plots"
mkdir -p "$RESULTS_DIR/uncertainty_distribution_plots"
mkdir -p "$RESULTS_DIR/summary_analysis"

# Check if required directories exist
if [ ! -d "old_vanilla_output" ]; then
    echo "Error: old_vanilla_output directory not found!"
    exit 1
fi

if [ ! -d "old_perturb_sampling_output" ]; then
    echo "Error: old_perturb_sampling_output directory not found!"
    exit 1
fi

if [ ! -d "dataset/TREA_dataset" ]; then
    echo "Error: dataset/TREA_dataset directory not found!"
    exit 1
fi

echo "Step 1: Running comprehensive metrics with all plots and visualizations..."
echo "----------------------------------------------------------------------"

# Run comprehensive analysis with results saved to final directory as CSV
python comprehensive_metrics_with_plots.py --all_models --all_tasks \
    --output "$RESULTS_DIR/comprehensive_metrics_results.csv" --abstention_step 0.1

if [ $? -ne 0 ]; then
    echo "Error: Comprehensive metrics analysis failed!"
    exit 1
fi

# Move generated plots to results directory
if [ -d "risk_coverage_plots" ]; then
    cp -r risk_coverage_plots/* "$RESULTS_DIR/risk_coverage_plots/"
    echo "  Copied risk coverage plots"
fi

if [ -d "ece_calibration_plots" ]; then
    cp -r ece_calibration_plots/* "$RESULTS_DIR/ece_calibration_plots/"
    echo "  Copied ECE calibration plots"
fi

if [ -d "uncertainty_distribution_plots" ]; then
    cp -r uncertainty_distribution_plots/* "$RESULTS_DIR/uncertainty_distribution_plots/"
    echo "  Copied uncertainty distribution plots"
fi

echo ""
echo "Step 2: Creating summary tables and aggregate visualizations..."
echo "------------------------------------------------------------"

# Install required packages if needed
pip install pandas seaborn > /dev/null 2>&1

# Create summary tables
python create_summary_tables.py \
    --results_file "$RESULTS_DIR/comprehensive_metrics_results.csv" \
    --output_dir "$RESULTS_DIR/summary_analysis"

if [ $? -ne 0 ]; then
    echo "Error: Summary table creation failed!"
    exit 1
fi

# Clean up temporary directories
rm -rf risk_coverage_plots ece_calibration_plots uncertainty_distribution_plots

echo ""
echo "======================================================================="
echo "ANALYSIS COMPLETE!"
echo "======================================================================="
echo ""
echo "All results have been saved to: $RESULTS_DIR"
echo ""
echo "Generated Files and Directories:"
echo "================================="
echo ""
echo "MAIN RESULTS:"
echo "  - comprehensive_metrics_results.csv (main results in CSV format)"
echo ""
echo "PLOTS AND VISUALIZATIONS:"
echo "  - risk_coverage_plots/ (risk-coverage and accuracy-coverage curves)"
echo "  - ece_calibration_plots/ (ECE calibration bar plots)" 
echo "  - uncertainty_distribution_plots/ (confidence distribution scatter plots)"
echo ""
echo "SUMMARY ANALYSIS:"
echo "  - summary_analysis/ (aggregated tables and comparison visualizations)"
echo "    - overall_model_summary.csv"
echo "    - task_wise_summary.csv"
echo "    - metric-specific summary files"
echo "    - comparison plots and heatmaps"
echo ""
echo "======================================================================="
echo "METRICS INCLUDED:"
echo "======================================================================="
echo "- Accuracy"
echo "- AUROC (Area Under ROC Curve)"  
echo "- AUPRC (Area Under Precision-Recall Curve)"
echo "- Brier Score"
echo "- ECE (Expected Calibration Error)"
echo "- AURC (Area Under Risk-Coverage Curve)"
echo "- Risk-Coverage Analysis with abstention tables"
echo "- Calibration plots with bin-wise analysis"
echo "- Confidence distribution visualizations"
echo ""
echo "Analysis completed successfully!"
