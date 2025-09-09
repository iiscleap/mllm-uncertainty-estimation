import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def create_summary_tables(results_file, output_dir="summary_tables"):
    """Create summary tables and visualizations from comprehensive results"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the results - handle both CSV and TSV
    if results_file.endswith('.csv'):
        df = pd.read_csv(results_file)
    else:
        df = pd.read_csv(results_file, sep='\t')
    
    print(f"Loaded results for {len(df)} model-dataset combinations")
    print(f"Models: {sorted(df['Model'].unique())}")
    print(f"Datasets: {sorted(df['Dataset'].unique())}")
    
    # Create pivot tables for each metric
    metrics = ['Accuracy', 'AUROC', 'AUPRC', 'Brier_Score', 'ECE', 'AURC']
    
    summary_tables = {}
    
    for metric in metrics:
        if metric in df.columns:
            pivot = df.pivot(index='Model', columns='Dataset', values=metric)
            summary_tables[metric] = pivot
            
            # Save to CSV
            csv_file = os.path.join(output_dir, f'{metric.lower()}_summary.csv')
            pivot.to_csv(csv_file)
            print(f"Saved {metric} summary to {csv_file}")
    
    # Create overall summary with means and std
    overall_summary = []
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        summary_row = {
            'Model': model,
            'Avg_Accuracy': model_data['Accuracy'].mean(),
            'Std_Accuracy': model_data['Accuracy'].std(),
            'Avg_AUROC': model_data['AUROC'].mean(),
            'Std_AUROC': model_data['AUROC'].std(),
            'Avg_AUPRC': model_data['AUPRC'].mean(),
            'Std_AUPRC': model_data['AUPRC'].std(),
            'Avg_Brier': model_data['Brier_Score'].mean(),
            'Std_Brier': model_data['Brier_Score'].std(),
            'Avg_ECE': model_data['ECE'].mean(),
            'Std_ECE': model_data['ECE'].std(),
            'Avg_AURC': model_data['AURC'].mean(),
            'Std_AURC': model_data['AURC'].std(),
            'Samples': int(model_data['Samples'].mean())
        }
        overall_summary.append(summary_row)
    
    overall_df = pd.DataFrame(overall_summary)
    overall_csv = os.path.join(output_dir, 'overall_model_summary.csv')
    overall_df.to_csv(overall_csv, index=False)
    print(f"Saved overall model summary to {overall_csv}")
    
    # Create dataset-wise summary
    dataset_summary = []
    for dataset in df['Dataset'].unique():
        dataset_data = df[df['Dataset'] == dataset]
        dataset_row = {
            'Dataset': dataset,
            'Avg_Accuracy': dataset_data['Accuracy'].mean(),
            'Std_Accuracy': dataset_data['Accuracy'].std(),
            'Avg_AUROC': dataset_data['AUROC'].mean(),
            'Std_AUROC': dataset_data['AUROC'].std(),
            'Avg_AUPRC': dataset_data['AUPRC'].mean(),
            'Std_AUPRC': dataset_data['AUPRC'].std(),
            'Avg_Brier': dataset_data['Brier_Score'].mean(),
            'Std_Brier': dataset_data['Brier_Score'].std(),
            'Avg_ECE': dataset_data['ECE'].mean(),
            'Std_ECE': dataset_data['ECE'].std(),
            'Avg_AURC': dataset_data['AURC'].mean(),
            'Std_AURC': dataset_data['AURC'].std(),
        }
        dataset_summary.append(dataset_row)
    
    dataset_df = pd.DataFrame(dataset_summary)
    dataset_csv = os.path.join(output_dir, 'dataset_wise_summary.csv')
    dataset_df.to_csv(dataset_csv, index=False)
    print(f"Saved dataset-wise summary to {dataset_csv}")
    
    # Create visualizations
    create_summary_visualizations(df, summary_tables, output_dir)
    
    return summary_tables, overall_df, dataset_df

def create_summary_visualizations(df, summary_tables, output_dir):
    """Create summary visualization plots"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Metric comparison across models
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    metrics = ['Accuracy', 'AUROC', 'AUPRC', 'Brier_Score', 'ECE', 'AURC']
    
    for i, metric in enumerate(metrics):
        if metric in summary_tables:
            summary_tables[metric].plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'{metric.replace("_", " ")} by Model and Dataset')
            axes[i].set_xlabel('Model')
            axes[i].set_ylabel(metric.replace('_', ' '))
            axes[i].legend(title='Dataset')
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved metrics comparison plot")
    
    # 2. Heatmap of AUROC scores
    if 'AUROC' in summary_tables:
        plt.figure(figsize=(10, 8))
        sns.heatmap(summary_tables['AUROC'], annot=True, cmap='YlOrRd', 
                    cbar_kws={'label': 'AUROC'}, fmt='.3f')
        plt.title('AUROC Heatmap: Models vs Datasets')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'auroc_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved AUROC heatmap")
    
    # 3. ECE vs AUROC scatter plot
    plt.figure(figsize=(10, 8))
    for dataset in df['Dataset'].unique():
        dataset_data = df[df['Dataset'] == dataset]
        plt.scatter(dataset_data['ECE'], dataset_data['AUROC'], 
                   label=f'Dataset: {dataset}', s=100, alpha=0.7)
    
    for i, row in df.iterrows():
        plt.annotate(row['Model'], (row['ECE'], row['AUROC']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('ECE (Expected Calibration Error)')
    plt.ylabel('AUROC')
    plt.title('Calibration vs Discrimination Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ece_vs_auroc_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved ECE vs AUROC scatter plot")
    
    # 4. Box plots for metric distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            df.boxplot(column=metric, by='Dataset', ax=axes[i])
            axes[i].set_title(f'{metric.replace("_", " ")} Distribution by Dataset')
            axes[i].set_xlabel('Dataset')
            axes[i].set_ylabel(metric.replace('_', ' '))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved metric distributions plot")
    
    # 5. Model ranking plot
    model_means = df.groupby('Model')[['AUROC', 'AUPRC', 'ECE', 'AURC']].mean()
    
    # Normalize metrics (lower is better for ECE and AURC, higher is better for AUROC and AUPRC)
    normalized_means = model_means.copy()
    normalized_means['ECE'] = 1 - normalized_means['ECE']  # Invert ECE
    normalized_means['AURC'] = 1 - normalized_means['AURC']  # Invert AURC
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(model_means.index))
    width = 0.2
    
    ax.bar(x - 1.5*width, normalized_means['AUROC'], width, label='AUROC', alpha=0.8)
    ax.bar(x - 0.5*width, normalized_means['AUPRC'], width, label='AUPRC', alpha=0.8)
    ax.bar(x + 0.5*width, normalized_means['ECE'], width, label='1-ECE', alpha=0.8)
    ax.bar(x + 1.5*width, normalized_means['AURC'], width, label='1-AURC', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Normalized Score (Higher is Better)')
    ax.set_title('Model Performance Comparison (All Metrics Normalized)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_means.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved model ranking plot")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, 
                       default="comprehensive_metrics_results.csv",
                       help="Path to the comprehensive results file (CSV or TSV)")
    parser.add_argument("--output_dir", type=str, default="summary_tables",
                       help="Output directory for summary tables and plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Results file not found: {args.results_file}")
        print("Please run the comprehensive metrics script first.")
        exit(1)
    
    print(f"Creating summary tables and visualizations...")
    print(f"Input file: {args.results_file}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    summary_tables, overall_df, dataset_df = create_summary_tables(args.results_file, args.output_dir)
    
    print("\n" + "=" * 50)
    print("OVERALL MODEL PERFORMANCE:")
    print("=" * 50)
    print(overall_df.round(4))
    
    print("\n" + "=" * 50)
    print("DATASET-WISE PERFORMANCE:")
    print("=" * 50)
    print(dataset_df.round(4))
    
    print(f"\nAll summary tables and visualizations saved to: {args.output_dir}/")
