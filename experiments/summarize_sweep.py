import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Summarize Gamma Sweep Results")
    parser.add_argument('--input', type=str, required=True, help='Path to the grid search results CSV')
    parser.add_argument('--output', type=str, required=True, help='Path to save the summary CSV')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    # Load results
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 유효한 메트릭 확인
    if 'valid_ndcg_100' in df.columns:
        metric = 'valid_ndcg_100'
    elif 'NDCG@100' in df.columns:
        metric = 'NDCG@100'
    else:
        print(f"Error: Required metrics not found in {args.input}")
        print(f"Available columns: {list(df.columns)}")
        return

    print(f"Summarizing {args.input} by best '{metric}' per alpha...")

    # Group by alpha and get the row with max metric
    # idxmax()는 그룹 내에서 해당 컬럼이 최대값인 인덱스를 반환
    idx = df.groupby('alpha')[metric].idxmax()
    summary_df = df.loc[idx].sort_values('alpha')

    # 저장할 컬럼 명시 (정확한 이름 사용)
    cols = ['alpha', 'reg_lambda', metric, 'Recall@20', 'Recall@50', 'NDCG@100', 'train_time']
    # 존재하는 컬럼만 필터링
    existing_cols = [c for c in cols if c in summary_df.columns]
    
    summary_df = summary_df[existing_cols]

    # 결과 저장
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary_df.to_csv(args.output, index=False)
    print(f"Successfully saved summary to: {args.output}")

if __name__ == "__main__":
    main()
