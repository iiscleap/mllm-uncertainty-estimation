#!/usr/bin/env python3
"""
SCRIPT 2: Table Formatting and Display
It reads the consolidated CSV files and creates beautifully formatted text tables
that exactly match the structure shown in the reference images.
"""
import pandas as pd
import os
import numpy as np

class TableFormatter:
    def __init__(self):
        self.metrics = ["Accuracy", "AUROC", "AUPRC", "Brier_Score", "ECE", "AURC"]
        
        # Model order from reference images
        self.vlm_model_order = ['Gemma-3', 'LLaVA-1.6', 'Qwen-2.5-VL', 'Phi-4', 'Pixtral']
        self.lalm_model_order = ['Qwen2-Audio', 'SALMONN', 'Capt. + Qwen2-7B']
        
        # Model name mappings
        self.vlm_model_mapping = {
            'gemma3': 'Gemma-3',
            'llava': 'LLaVA-1.6', 
            'qwenvl': 'Qwen-2.5-VL',
            'phi4': 'Phi-4',
            'pixtral': 'Pixtral'
        }
        
        self.lalm_model_mapping = {
            'qwen': 'Qwen2-Audio',
            'salmonn': 'SALMONN',
            'desc_llm': 'Capt. + Qwen2-7B'
        }
        
        # Dataset information
        self.vlm_datasets = ['blink', 'vsr']
        self.vlm_dataset_names = ['BLINK', 'VSR']
        
        self.lalm_datasets = ['order', 'duration', 'count']
        self.lalm_dataset_names = ['TREA-O', 'TREA-D', 'TREA-C']
        
        # Baseline column orders
        self.vlm_baseline_cols = ['OE', 'VB', 'IA-I', 'IA-T', 'IA-IT', 'RU', 'BU']
        self.lalm_baseline_cols = ['OE', 'VB', 'IA-A', 'IA-T', 'IA-AT', 'RU', 'BU']

    def format_value(self, val, is_accuracy=False):
        """Format numerical values for display"""
        if pd.isna(val) or val == '':
            return '—'
        try:
            val = float(val)
            return f"{val:.2f}"
        except:
            return '—'

    def load_consolidated_data(self, metric, table_type):
        """Load consolidated data for a specific metric and table type"""
        data_dir = "consolidated_data"
        file_path = os.path.join(data_dir, f"{metric}_{table_type}_consolidated.csv")
        
        if not os.path.exists(file_path):
            # Fallback to paper_ready_tables
            data_dir = "paper_ready_tables"
            file_path = os.path.join(data_dir, f"{metric}_{table_type}.csv")
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            return df
        else:
            print(f"File not found: {file_path}")
            return pd.DataFrame()

    def create_vlm_table(self, metric="AUROC"):
        """Create VLM table with exact formatting from reference image"""
        # Load data
        df = self.load_consolidated_data(metric, "VLM")
        acc_df = self.load_consolidated_data("Accuracy", "VLM")
        
        if df.empty:
            print(f"No VLM data found for {metric}")
            return ""
        
        # Merge with accuracy data for prediction accuracy column
        if not acc_df.empty:
            merged_df = pd.merge(df, acc_df[['Model', 'Dataset', 'OE']], 
                                on=['Model', 'Dataset'], suffixes=('', '_acc'))
            merged_df['Pred_Acc'] = merged_df['OE_acc']
            merged_df = merged_df.drop('OE_acc', axis=1)
        else:
            merged_df = df.copy()
            merged_df['Pred_Acc'] = np.nan
        
        table_lines = []
        
        # Main header line
        table_lines.append("─" * 120)
        
        # Table headers
        header1 = f"{'Dataset Model':<20}{'Pred.':<10}{'Baseline Results (' + metric + ')':<56}{'Ours (' + metric + ')':<15}"
        table_lines.append(header1)
        
        # Sub-headers
        subheader = f"{'':20}{'Acc.':<10}"
        for col in self.vlm_baseline_cols:
            subheader += f"{col:<8}"
        subheader += f"{'FESTA':<15}"
        table_lines.append(subheader)
        
        table_lines.append("─" * 120)
        
        # Process each dataset
        for dataset, dataset_display in zip(self.vlm_datasets, self.vlm_dataset_names):
            dataset_data = merged_df[merged_df['Dataset'].str.lower() == dataset.lower()]
            
            if len(dataset_data) == 0:
                continue
            
            # Sort models according to reference image order
            dataset_data_sorted = []
            for model_name in self.vlm_model_order:
                model_key = [k for k, v in self.vlm_model_mapping.items() if v == model_name]
                if model_key:
                    model_rows = dataset_data[dataset_data['Model'] == model_key[0]]
                    if len(model_rows) > 0:
                        dataset_data_sorted.append((model_name, model_rows.iloc[0]))
            
            # Add rows for this dataset
            for i, (model_display, row) in enumerate(dataset_data_sorted):
                if i == 0:  # First row of dataset
                    line = f"{dataset_display:<8}{model_display:<12}"
                else:
                    line = f"{'':8}{model_display:<12}"
                
                # Add prediction accuracy
                line += f"{self.format_value(row['Pred_Acc'], True):<10}"
                
                # Add baseline results
                for col in self.vlm_baseline_cols:
                    if col in row and not pd.isna(row[col]):
                        line += f"{self.format_value(row[col]):<8}"
                    else:
                        line += f"{'—':<8}"
                
                # Add FESTA results
                if 'FESTA' in row and not pd.isna(row['FESTA']):
                    line += f"**{self.format_value(row['FESTA'])}**"
                else:
                    line += f"{'—':<15}"
                
                table_lines.append(line)
            
            # Add average row
            if dataset_data_sorted:
                avg_pred_acc = np.mean([row['Pred_Acc'] for _, row in dataset_data_sorted if not pd.isna(row['Pred_Acc'])])
                avg_line = f"{'':8}{'Avg.':<12}{self.format_value(avg_pred_acc, True):<10}"
                
                # Calculate averages for baseline methods
                for col in self.vlm_baseline_cols:
                    col_vals = [row[col] for _, row in dataset_data_sorted if col in row and not pd.isna(row[col])]
                    if col_vals:
                        avg_val = np.mean(col_vals)
                        avg_line += f"{self.format_value(avg_val):<8}"
                    else:
                        avg_line += f"{'—':<8}"
                
                # Calculate FESTA average
                festa_vals = [row['FESTA'] for _, row in dataset_data_sorted if 'FESTA' in row and not pd.isna(row['FESTA'])]
                if festa_vals:
                    avg_festa = np.mean(festa_vals)
                    avg_line += f"**{self.format_value(avg_festa)}**"
                else:
                    avg_line += f"{'—':<15}"
                
                table_lines.append(avg_line)
            
            # Add separator between datasets
            if dataset != self.vlm_datasets[-1]:
                table_lines.append("")
        
        table_lines.append("─" * 120)
        
        return "\n".join(table_lines)

    def create_lalm_table(self, metric="AUROC"):
        """Create LALM table with exact formatting from reference image"""
        # Load data
        df = self.load_consolidated_data(metric, "LALM")
        acc_df = self.load_consolidated_data("Accuracy", "LALM")
        
        if df.empty:
            print(f"No LALM data found for {metric}")
            return ""
        
        # Merge with accuracy data for prediction accuracy column
        if not acc_df.empty:
            merged_df = pd.merge(df, acc_df[['Model', 'Dataset', 'OE']], 
                                on=['Model', 'Dataset'], suffixes=('', '_acc'))
            merged_df['Pred_Acc'] = merged_df['OE_acc']
            merged_df = merged_df.drop('OE_acc', axis=1)
        else:
            merged_df = df.copy()
            merged_df['Pred_Acc'] = np.nan
        
        table_lines = []
        
        # Main header line
        table_lines.append("─" * 124)
        
        # Table headers
        header1 = f"{'Dataset Model':<20}{'Pred.':<10}{'Baseline Results (' + metric + ')':<64}{'Ours (' + metric + ')':<15}"
        table_lines.append(header1)
        
        # Sub-headers
        subheader = f"{'':20}{'Acc.':<10}"
        for col in self.lalm_baseline_cols:
            subheader += f"{col:<8}"
        subheader += f"{'FESTA':<15}"
        table_lines.append(subheader)
        
        table_lines.append("─" * 124)
        
        # Process each task/dataset
        for task, task_display in zip(self.lalm_datasets, self.lalm_dataset_names):
            task_data = merged_df[merged_df['Dataset'].str.lower() == task.lower()]
            
            if len(task_data) == 0:
                continue
            
            # Sort models according to reference image order
            task_data_sorted = []
            for model_name in self.lalm_model_order:
                model_key = [k for k, v in self.lalm_model_mapping.items() if v == model_name]
                if model_key:
                    model_rows = task_data[task_data['Model'] == model_key[0]]
                    if len(model_rows) > 0:
                        task_data_sorted.append((model_name, model_rows.iloc[0]))
            
            # Add rows for this task
            for i, (model_display, row) in enumerate(task_data_sorted):
                if i == 0:  # First row of task
                    line = f"{task_display:<8}{model_display:<12}"
                else:
                    line = f"{'':8}{model_display:<12}"
                
                # Add prediction accuracy
                line += f"{self.format_value(row['Pred_Acc'], True):<10}"
                
                # Add baseline results
                for col in self.lalm_baseline_cols:
                    if col in row and not pd.isna(row[col]):
                        line += f"{self.format_value(row[col]):<8}"
                    else:
                        line += f"{'—':<8}"
                
                # Add FESTA results
                if 'FESTA' in row and not pd.isna(row['FESTA']):
                    line += f"**{self.format_value(row['FESTA'])}**"
                else:
                    line += f"{'—':<15}"
                
                table_lines.append(line)
            
            # Add average row
            if task_data_sorted:
                avg_pred_acc = np.mean([row['Pred_Acc'] for _, row in task_data_sorted if not pd.isna(row['Pred_Acc'])])
                avg_line = f"{'':8}{'Avg.':<12}{self.format_value(avg_pred_acc, True):<10}"
                
                # Calculate averages for baseline methods
                for col in self.lalm_baseline_cols:
                    col_vals = [row[col] for _, row in task_data_sorted if col in row and not pd.isna(row[col])]
                    if col_vals:
                        avg_val = np.mean(col_vals)
                        avg_line += f"{self.format_value(avg_val):<8}"
                    else:
                        avg_line += f"{'—':<8}"
                
                # Calculate FESTA average
                festa_vals = [row['FESTA'] for _, row in task_data_sorted if 'FESTA' in row and not pd.isna(row['FESTA'])]
                if festa_vals:
                    avg_festa = np.mean(festa_vals)
                    avg_line += f"**{self.format_value(avg_festa)}**"
                else:
                    avg_line += f"{'—':<15}"
                
                table_lines.append(avg_line)
            
            # Add separator between tasks
            if task != self.lalm_datasets[-1]:
                table_lines.append("")
        
        table_lines.append("─" * 124)
        
        return "\n".join(table_lines)

    def create_all_formatted_tables(self):
        """Create all formatted tables for all metrics"""
        output_dir = "final_formatted_tables"
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating formatted tables...")
        
        for metric in self.metrics:
            print(f"Creating {metric} tables...")
            
            # Create VLM table
            try:
                vlm_table = self.create_vlm_table(metric)
                if vlm_table:
                    with open(f"{output_dir}/{metric}_VLM_formatted.txt", 'w') as f:
                        f.write(vlm_table)
                    print(f"Created: {metric}_VLM_formatted.txt")
            except Exception as e:
                print(f"Error creating VLM {metric} table: {e}")
            
            # Create LALM table
            try:
                lalm_table = self.create_lalm_table(metric)
                if lalm_table:
                    with open(f"{output_dir}/{metric}_LALM_formatted.txt", 'w') as f:
                        f.write(lalm_table)
                    print(f"Created: {metric}_LALM_formatted.txt")
            except Exception as e:
                print(f"Error creating LALM {metric} table: {e}")
        
        print(f"All formatted tables saved to {output_dir}/")

    def run_formatting(self):
        """Main method to run the complete table formatting process"""
        print("Starting table formatting...")
        
        # Check if consolidated data exists
        if not os.path.exists("consolidated_data") and not os.path.exists("paper_ready_tables"):
            print("No consolidated data found! Please run script 01 first.")
            return
        
        # Create formatted tables
        self.create_all_formatted_tables()
        
        print("Table formatting completed successfully!")
        print("Generated directory:")
        print("- final_formatted_tables/ (publication-ready tables)")

def main():
    """Main function to run table formatting"""
    formatter = TableFormatter()
    formatter.run_formatting()

if __name__ == "__main__":
    main()
