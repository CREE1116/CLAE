import pandas as pd
import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Summarize Gamma Sweep Results")
    parser.add_argument('--input', type=str, default='exp_result/grid_search_ASPIRE_EASE_steam_strong.csv', help='Path to the grid search results CSV')
    parser.add_argument('--output', type=str, default='exp_result/summary_ASPIRE_EASE_gamma_sweep.csv', help='Path to save the summary CSV')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    # Load results
    df = pd.read_csv(args.input)

    # Sort and group by alpha (gamma)
    # We want to find the row with the maximum validation NDCG for each alpha
    if 'valid_ndcg_100' not in df.columns:
        # Fallback to test metric if valid metric is missing for some reason
        metric = 'NDCG@100'
    else:
        metric = 'valid_ndcg_100'

    print(f"Summarizing by grouping 'alpha' and finding best '{metric}'...")

    # Group by alpha and get the row index with max metric
    idx = df.groupby('alpha')[metric].idxmax()
    summary_df = df.loc[idx].sort_values('alpha')

    # Keep relevant columns and move alpha to front
    cols = ['alpha', 'reg_lambda', metric, 'Recall@20', 'Recall@50', 'NDCG@100', 'train_time']
    # Filter only existing columns
    cols = [c for c in cols if c in summary_df.columns]
    
    summary_df = summary_df[cols]

    # Save to CSV
    summary_df.to_csv(args.output, index=False)
    print(f"Summary saved to {args.output}")
    print("\nBest results per Gamma (alpha):")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
