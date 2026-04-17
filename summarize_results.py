import os
import pandas as pd
import glob
import ast

def summarize_results():
    results_dir = 'results'
    # Pattern to match grid_search files: grid_search_<model>_<dataset>_<setting>.csv
    # e.g., grid_search_CLAE_steam_strong.csv
    file_pattern = os.path.join(results_dir, 'grid_search_*.csv')
    files = glob.glob(file_pattern)
    
    # Group files by (dataset, setting)
    data_groups = {}
    for f in files:
        basename = os.path.basename(f)
        # We need to be careful with models that have underscores, like EASE_DAN or IPS_LAE
        # The filename format seems to be: grid_search_<MODEL>_<DATASET>_<SETTING>.csv
        # Let's try to parse from the end.
        parts = basename.replace('.csv', '').split('_')
        # Parts: ['grid', 'search', MODEL..., DATASET, SETTING]
        # Since DATASET and SETTING are usually single words without underscores (e.g., steam, strong, ml-20m, netflix)
        setting = parts[-1]
        dataset = parts[-2]
        model = '_'.join(parts[2:-2])
        
        group_key = (dataset, setting)
        if group_key not in data_groups:
            data_groups[group_key] = []
        data_groups[group_key].append((model, f))

    for (dataset, setting), model_files in data_groups.items():
        summary_rows = []
        for model, f in model_files:
            try:
                df = pd.read_csv(f)
                if df.empty:
                    continue
                
                # Identify parameter columns
                # Based on observation: model, dataset, [params], train_time, valid_ndcg_100, [results]
                cols = list(df.columns)
                try:
                    ds_idx = cols.index('dataset')
                    tt_idx = cols.index('train_time')
                    param_cols = cols[ds_idx + 1:tt_idx]
                except ValueError:
                    # Fallback if train_time or dataset is missing
                    # Just take anything before valid_ndcg_100 that isn't model or dataset
                    v_idx = cols.index('valid_ndcg_100')
                    param_cols = [c for c in cols[:v_idx] if c not in ['model', 'dataset']]

                # Find max valid_ndcg_100
                max_ndcg = df['valid_ndcg_100'].max()
                best_rows = df[df['valid_ndcg_100'] == max_ndcg].copy()
                
                if len(best_rows) > 1:
                    # Calculate average of numeric parameters
                    numeric_params = []
                    for col in param_cols:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            numeric_params.append(col)
                    
                    if numeric_params:
                        best_rows['param_avg'] = best_rows[numeric_params].mean(axis=1)
                        best_row = best_rows.sort_values(by='param_avg', ascending=False).iloc[0]
                    else:
                        best_row = best_rows.iloc[0]
                else:
                    best_row = best_rows.iloc[0]
                
                # Prepare row for summary
                # Required columns: model, valid_ndcg_100, NDCG@10, Recall@10, uNDCG@10, ... best_params
                result_cols = ['model', 'valid_ndcg_100', 'NDCG@10', 'Recall@10', 'uNDCG@10', 
                               'NDCG@20', 'Recall@20', 'uNDCG@20', 'NDCG@50', 'Recall@50', 'uNDCG@50', 
                               'NDCG@100', 'Recall@100', 'uNDCG@100']
                
                row_data = {col: best_row[col] for col in result_cols if col in best_row}
                
                # Format best_params
                params_dict = {col: best_row[col] for col in param_cols}
                row_data['best_params'] = str(params_dict)
                
                summary_rows.append(row_data)
                
            except Exception as e:
                print(f"Error processing {f}: {e}")

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            # Sort by valid_ndcg_100 descending
            summary_df = summary_df.sort_values(by='valid_ndcg_100', ascending=False)
            
            output_file = os.path.join(results_dir, f"{dataset}_{setting}_comparison.csv")
            summary_df.to_csv(output_file, index=False)
            print(f"Saved summary to {output_file}")

if __name__ == "__main__":
    summarize_results()
