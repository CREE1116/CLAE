import pandas as pd
import os

# 파일 경로 리스트
files = [
    '/Users/leejongmin/code/CLAE/results/grid_search_CLAE_steam_strong.csv',
    '/Users/leejongmin/code/CLAE/results/grid_search_EASE_DAN_steam_strong.csv',
    '/Users/leejongmin/code/CLAE/results/grid_search_EASE_steam_strong.csv',
    '/Users/leejongmin/code/CLAE/results/grid_search_GFCF_steam_strong.csv',
    '/Users/leejongmin/code/CLAE/results/grid_search_IPS_LAE_steam_strong.csv',
    '/Users/leejongmin/code/CLAE/results/grid_search_RLAE_steam_strong.csv'
]

# 메트릭 키워드 (이것들이 포함된 컬럼은 지표로 간주)
metric_keywords = ['NDCG', 'Recall', 'valid_ndcg_100']

best_results = []

for file_path in files:
    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}")
        continue
        
    df = pd.read_csv(file_path)
    
    if 'valid_ndcg_100' not in df.columns:
        print(f"Warning: 'valid_ndcg_100' column missing in {file_path}")
        continue

    # valid_ndcg_100 기준으로 최적의 행 선택
    best_row = df.loc[df['valid_ndcg_100'].idxmax()].copy()
    
    # 지표가 아닌 컬럼들 (파라미터 추출을 위해)
    # model, dataset, train_time 등은 제외
    non_param_cols = ['model', 'dataset', 'train_time'] + [c for c in df.columns if any(k in c for k in metric_keywords)]
    
    # 파라미터 정보 추출
    params = {k: v for k, v in best_row.to_dict().items() if k not in non_param_cols}
    best_row['best_params'] = str(params)
    
    best_results.append(best_row)

# 결과 데이터프레임 생성 및 정렬
comparison_df = pd.DataFrame(best_results)
comparison_df = comparison_df.sort_values(by='valid_ndcg_100', ascending=False)

# 모든 지표 컬럼 찾기 (정렬을 위해)
all_cols = comparison_df.columns.tolist()
metrics = [c for c in all_cols if any(k in c for k in metric_keywords)]

# 지표들을 보기 좋게 정렬 (10, 20, 50, 100 순서)
def sort_key(col):
    if 'valid' in col: return (0, 0)
    import re
    match = re.search(r'@(\d+)', col)
    k_val = int(match.group(1)) if match else 999
    
    type_priority = 0
    if 'uNDCG' in col: type_priority = 3
    elif 'NDCG' in col: type_priority = 1
    elif 'Recall' in col: type_priority = 2
    
    return (k_val, type_priority)

sorted_metrics = sorted(metrics, key=sort_key)

# 최종 출력 컬럼 구성
cols_to_show = ['model'] + sorted_metrics + ['best_params']

print("\n=== Steam Strong Dataset Grid Search Comparison (All Metrics) ===")
# 터미널 가독성을 위해 상위 몇개 지표만 출력 (CSV에는 다 저장)
display_cols = ['model', 'valid_ndcg_100', 'NDCG@100', 'Recall@100', 'NDCG(head)@100', 'NDCG(tail)@100']
print(comparison_df[display_cols].to_string(index=False))

# CSV 파일로 저장 (모든 지표 포함)
output_path = '/Users/leejongmin/code/CLAE/results/steam_strong_comparison.csv'
comparison_df[cols_to_show].to_csv(output_path, index=False)
print(f"\nAll metrics and results saved to: {output_path}")
