#!/usr/bin/env python3
"""
SCRIPT 1: Data Extraction and Consolidation
Extracts data from baseline CSV files and FESTA results,
creates consolidated CSV files for the table formatting script.
"""
import pandas as pd
import os
from pathlib import Path
import numpy as np

class DataExtractor:
    def __init__(self):
        self.metrics = ["Accuracy", "AUROC", "AUPRC", "Brier_Score", "ECE", "AURC"]
        self.vlm_datasets = ['blink', 'vsr'] 
        self.lalm_datasets = ['count', 'duration', 'order']
        
    def extract_baseline_data(self):
        """Extract data from all baseline comprehensive CSV files"""
        # Define the CSV file paths with their method abbreviations
        csv_files = [
            ("VB", "verbalized_baseline/lalm/comprehensive_results/comprehensive_metrics.csv"),
            ("VB", "verbalized_baseline/lvlm/comprehensive_results/comprehensive_metrics.csv"),
            ("OE", "output_entropy_baseline/lalm/comprehensive_results/comprehensive_metrics.csv"),
            ("OE", "output_entropy_baseline/lvlm/comprehensive_results/comprehensive_metrics.csv"),
            ("RU", "rephrase_uncertainty/lalm/comprehensive_results/comprehensive_metrics.csv"),
            ("RU", "rephrase_uncertainty/lvlm/comprehensive_results/comprehensive_metrics.csv"),
            ("BU", "blackbox_uncertainty/prompting_sampling_baseline_lalm/comprehensive_results/comprehensive_metrics.csv"),
            ("BU", "blackbox_uncertainty/prompting_sampling_baseline_vlm/comprehensive_results/comprehensive_metrics.csv"),
            ("IA-A", "bahat_baseline_lalm/comprehensive_results/comprehensive_metrics.csv"),
            ("IA-T", "bahat_baseline_vlm/comprehensive_results/comprehensive_metrics.csv")
        ]
        
        # Collect all data
        all_data = []
        
        # Process each CSV file
        for method_abbrev, file_path in csv_files:
            if os.path.exists(file_path):
                print(f"Processing baseline: {file_path}")
                df = pd.read_csv(file_path)
                
                for _, row in df.iterrows():
                    # Determine dataset/task name
                    dataset = row.get('Dataset', row.get('Task', ''))
                    model = row.get('Model', '')
                    method = row.get('Method', '')
                    exp_type = row.get('Exp_Type', '')
                    
                    # Create method column name based on experiment details
                    if method_abbrev == "IA-A":  # bahat_baseline_lalm
                        if exp_type == "audio_only":
                            method_col = "IA-A"
                        elif exp_type == "text_only":
                            method_col = "IA-T"
                        elif exp_type == "text_audio":
                            method_col = "IA-AT"
                        else:
                            method_col = method_abbrev
                    elif method_abbrev == "IA-T":  # bahat_baseline_vlm
                        if exp_type == "image_only":
                            method_col = "IA-I"
                        elif exp_type == "text_only":
                            method_col = "IA-T"
                        elif exp_type == "text_image":
                            method_col = "IA-IT"
                        else:
                            method_col = method_abbrev
                    elif method_abbrev == "OE":
                        if method == "sampling" and exp_type == "orig":
                            method_col = "OE"
                        else:
                            continue  # Skip other OE variants
                    elif method_abbrev == "RU":
                        if exp_type == "text_only":
                            method_col = "RU"
                        else:
                            continue  # Skip other RU variants
                    else:
                        method_col = method_abbrev
                    
                    # Create record
                    record = {
                        'Model': model,
                        'Dataset': dataset,
                        'Method': method_col,
                        'Accuracy': row.get('Accuracy', ''),
                        'AUROC': row.get('AUROC', ''),
                        'AUPRC': row.get('AUPRC', ''),
                        'Brier_Score': row.get('Brier_Score', ''),
                        'ECE': row.get('ECE', ''),
                        'AURC': row.get('AURC', ''),
                        'Samples': row.get('Samples', '')
                    }
                    all_data.append(record)
            else:
                print(f"File not found: {file_path}")
        
        return pd.DataFrame(all_data)

    def load_festa_results(self):
        """Load FESTA results from the results directory"""
        festa_results = {}
        
        # Load LALM FESTA results
        lalm_file = "path/to/FESTA/results/lalm/comprehensive_metrics_results.csv"
        if os.path.exists(lalm_file):
            lalm_df = pd.read_csv(lalm_file)
            for _, row in lalm_df.iterrows():
                key = f"{row['Model']}_{row['Task']}"
                festa_results[key] = row
            print(f"Loaded {len(lalm_df)} LALM FESTA results")
        else:
            print(f"LALM FESTA results file not found: {lalm_file}")
        
        # Load LVLM FESTA results  
        lvlm_file = "path/to/FESTA/results/lvlm/comprehensive_metrics_results.csv"
        if os.path.exists(lvlm_file):
            lvlm_df = pd.read_csv(lvlm_file)
            for _, row in lvlm_df.iterrows():
                key = f"{row['Model']}_{row['Dataset']}"
                festa_results[key] = row
            print(f"Loaded {len(lvlm_df)} LVLM FESTA results")
        else:
            print(f"LVLM FESTA results file not found: {lvlm_file}")
        
        return festa_results

    def create_pivot_tables(self, baseline_df, festa_results):
        """Create consolidated pivot tables for each metric (only what's needed for table formatting)"""
        
        # Create output directory - only consolidated_data is needed for table formatting
        consolidated_dir = "consolidated_data"
        os.makedirs(consolidated_dir, exist_ok=True)
        
        # Process each metric
        for metric in self.metrics:
            print(f"Processing {metric}...")
            
            # Create pivot table for metric data
            metric_data = baseline_df[['Model', 'Dataset', 'Method', metric]].copy()
            metric_data = metric_data[metric_data[metric] != '']  # Remove empty values
            
            if not metric_data.empty:
                # Convert to numeric
                metric_data[metric] = pd.to_numeric(metric_data[metric], errors='coerce')
                
                # Create pivot table
                pivot_table = metric_data.pivot_table(
                    index=['Model', 'Dataset'], 
                    columns='Method', 
                    values=metric, 
                    aggfunc='first'
                ).round(4).reset_index()
                
                # Process VLM data
                vlm_data = pivot_table[pivot_table['Dataset'].isin(self.vlm_datasets)].copy()
                if not vlm_data.empty:
                    # Add FESTA results for VLM
                    vlm_data['FESTA'] = np.nan
                    for idx, row in vlm_data.iterrows():
                        festa_key = f"{row['Model']}_{row['Dataset']}"
                        if festa_key in festa_results:
                            vlm_data.at[idx, 'FESTA'] = festa_results[festa_key][metric]
                    
                    # Reorder columns for VLM
                    vlm_columns = ['Model', 'Dataset']
                    available_methods = [col for col in ['OE', 'VB', 'IA-I', 'IA-T', 'IA-IT', 'RU', 'BU', 'FESTA'] if col in vlm_data.columns]
                    vlm_columns.extend(available_methods)
                    vlm_data = vlm_data[vlm_columns].round(4)
                    
                    # Save consolidated VLM table (only output needed)
                    consolidated_vlm_output = os.path.join(consolidated_dir, f"{metric}_VLM_consolidated.csv")
                    vlm_data.to_csv(consolidated_vlm_output, index=False)
                
                # Process LALM data
                lalm_data = pivot_table[pivot_table['Dataset'].isin(self.lalm_datasets)].copy()
                if not lalm_data.empty:
                    # Add FESTA results for LALM
                    lalm_data['FESTA'] = np.nan
                    for idx, row in lalm_data.iterrows():
                        festa_key = f"{row['Model']}_{row['Dataset']}"
                        if festa_key in festa_results:
                            lalm_data.at[idx, 'FESTA'] = festa_results[festa_key][metric]
                    
                    # Reorder columns for LALM
                    lalm_columns = ['Model', 'Dataset']
                    available_methods = [col for col in ['OE', 'VB', 'IA-A', 'IA-T', 'IA-AT', 'RU', 'BU', 'FESTA'] if col in lalm_data.columns]
                    lalm_columns.extend(available_methods)
                    lalm_data = lalm_data[lalm_columns].round(4)
                    
                    # Save consolidated LALM table (only output needed)
                    consolidated_lalm_output = os.path.join(consolidated_dir, f"{metric}_LALM_consolidated.csv")
                    lalm_data.to_csv(consolidated_lalm_output, index=False)
                
                print(f"Created tables for {metric}")

    def run_extraction(self):
        """Main method to run the complete data extraction process"""
        print("Starting data extraction and consolidation...")
        
        # Extract baseline data
        print("Extracting baseline data...")
        baseline_df = self.extract_baseline_data()
        
        if baseline_df.empty:
            print("No baseline data found!")
            return
        
        print(f"Total baseline records: {len(baseline_df)}")
        print(f"Unique methods: {baseline_df['Method'].unique()}")
        print(f"Unique models: {baseline_df['Model'].unique()}")
        print(f"Unique datasets: {baseline_df['Dataset'].unique()}")
        
        # Load FESTA results
        print("Loading FESTA results...")
        festa_results = self.load_festa_results()
        
        # Create pivot tables
        print("Creating pivot tables...")
        self.create_pivot_tables(baseline_df, festa_results)
        
        print("Data extraction and consolidation completed successfully!")
        print("Generated directory:")
        print("- consolidated_data/ (consolidated tables ready for formatting)")

def main():
    """Main function to run data extraction"""
    extractor = DataExtractor()
    extractor.run_extraction()

if __name__ == "__main__":
    main()

