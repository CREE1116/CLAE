import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    input_file = 'exp_result/summary_ASPIRE_EASE_gamma_sweep.csv'
    output_image = 'exp_result/gamma_sweep_plot.png'

    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    # Load summary results
    df = pd.read_csv(input_file)
    
    # Sort by alpha just in case
    df = df.sort_values('alpha')

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot multiple metrics
    metrics_to_plot = ['valid_ndcg_100', 'Recall@20', 'NDCG@100']
    markers = ['o', 's', '^']
    
    for metric, marker in zip(metrics_to_plot, markers):
        if metric in df.columns:
            plt.plot(df['alpha'], df[metric], marker=marker, label=metric)

    plt.title('ASPIRE-EASE Gamma (Alpha) Sweep Results', fontsize=14)
    plt.xlabel('Gamma (Alpha)', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Plot saved successfully to {output_image}")

if __name__ == "__main__":
    main()
