import subprocess
import os
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import re
import argparse
import sys

def generate_range(start, end, num, scale='linear'):
    """
    그리드 범위를 생성합니다.
    - linear: 선형 구간
    - log: 로그 스케일 구간 (start, end는 10의 거듭제곱이 아닌 실제 값 입력)
    """
    if num <= 1:
        return [start]
    
    if scale == 'log':
        # 로그 스케일 생성 (np.logspace 사용)
        return np.logspace(np.log10(start), np.log10(end), num=num).tolist()
    else:
        # 선형 스케일 생성 (np.linspace 사용)
        return np.linspace(start, end, num=num).tolist()

def parse_results(output):
    """
    Procedure.py의 print(results) 출력문을 파싱합니다.
    """
    match = re.search(r"\{'precision':.*\}", output, re.DOTALL)
    if not match:
        return None
    
    res_str = match.group(0)
    res_str = res_str.replace('array(', '').replace(')', '')
    try:
        # eval 환경에서 numpy array를 처리하기 위해 np 정의
        results = eval(res_str, {"np": np, "array": np.array})
        return results
    except Exception as e:
        print(f"Error parsing results: {e}")
        return None

def run_experiment(model, dataset, params, gpu, is_strong=True):
    # strong 또는 weak 디렉토리 결정
    base_dir = "strong" if is_strong else "weak"
    exec_dir = os.path.join(base_dir, "code")
    
    # 작업 디렉토리를 변경하므로 script_path는 main.py가 됩니다.
    cmd = [
        sys.executable, "main.py",
        "--model", model,
        "--dataset", dataset,
        "--gpu", str(gpu)
    ]
    
    for k, v in params.items():
        cmd.extend([f"--{k}", f"{v:.6f}" if isinstance(v, float) else str(v)])
    
    print(f"\nRunning: {' '.join(cmd)} in {exec_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=exec_dir)
    
    if result.returncode != 0:
        print(f"Error running {model}: {result.stderr}")
        return None
    
    parsed = parse_results(result.stdout)
    if parsed:
        flat_results = {"model": model, "dataset": dataset}
        flat_results.update(params)
        
        time_match = re.search(r"Training time: (\d+), Valid NDCG@100: ([\d.]+)", result.stdout)
        if time_match:
            flat_results["train_time"] = int(time_match.group(1))
            flat_results["valid_ndcg_100"] = float(time_match.group(2))
            
        topks = [10, 20, 50, 100]
        for i, k in enumerate(topks):
            if i < len(parsed['ndcg']):
                flat_results[f"NDCG@{k}"] = parsed['ndcg'][i]
                flat_results[f"Recall@{k}"] = parsed['recall'][i]
                flat_results[f"uNDCG@{k}"] = parsed['undcg'][i]
        
        return flat_results
    return None

def is_already_done(existing_df, params):
    if existing_df is None or existing_df.empty:
        return False
    
    # Check if a row with the same parameters exists
    mask = pd.Series([True] * len(existing_df))
    for k, v in params.items():
        if k in existing_df.columns:
            if isinstance(v, float):
                # Use a small epsilon for float comparison
                mask &= (existing_df[k] - v).abs() < 1e-6
            else:
                mask &= (existing_df[k] == v)
        else:
            return False # Column missing, treat as not done
    return mask.any()

def main():
    parser = argparse.ArgumentParser(description="Grid Search for CLAE/DCLAE")
    
    MODEL_LIST = [
        'EASE', 'RLAE', 'DLAE', 'LAE', 
        'DAN_EASE', 'DAN_RLAE', 'DAN_DLAE', 'DAN_LAE', 
        'ASPIRE_RLAE', 'ASPIRE_EASE', 'ASPIRE_DLAE', 'ASPIRE_LAE',
        # Others
        'CLAE', 'DCLAE', 'GFCF', 'RDLAE', 'EDLAE', 'EASE_DAN', 'IPS_LAE'
    ]
    
    parser.add_argument('--model', type=str, default='EASE', choices=MODEL_LIST)
    parser.add_argument('--dataset', type=str, default='yelp2018')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default='strong', choices=['strong', 'weak'])
    
    # 그리드 설정 인자들
    parser.add_argument('--reg_lambda_grid', nargs=4, metavar=('START', 'END', 'NUM', 'SCALE'), 
                        default=[0.1, 100.0, 5, 'log'], help='Grid for reg_lambda')
    parser.add_argument('--alpha_grid', nargs=4, metavar=('START', 'END', 'NUM', 'SCALE'), 
                        default=[0.0, 1.0, 3, 'linear'], help='Grid for alpha')
    parser.add_argument('--beta_grid', nargs=4, metavar=('START', 'END', 'NUM', 'SCALE'), 
                        default=[0.0, 1.0, 3, 'linear'], help='Grid for beta')
    parser.add_argument('--dropout_grid', nargs=4, metavar=('START', 'END', 'NUM', 'SCALE'), 
                        default=[0.1, 0.7, 4, 'linear'], help='Grid for dropout_p')
    parser.add_argument('--reg_p_grid', nargs=4, metavar=('START', 'END', 'NUM', 'SCALE'), 
                        default=[10, 1000, 5, 'log'], help='Grid for reg_p')
    parser.add_argument('--xi_grid', nargs=4, metavar=('START', 'END', 'NUM', 'SCALE'), 
                        default=[0.0, 0.9, 5, 'linear'], help='Grid for xi')
    parser.add_argument('--wbeta_grid', nargs=4, metavar=('START', 'END', 'NUM', 'SCALE'), 
                        default=[0.1, 1.0, 5, 'linear'], help='Grid for wbeta')
    parser.add_argument('--wtype', type=str, default='logsigmoid', help='Fixed wtype for IPS_LAE')
    
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    save_path = f"results/grid_search_{args.model}_{args.dataset}_{args.mode}.csv"
    
    # Load existing results if file exists
    existing_df = None
    if os.path.exists(save_path):
        try:
            existing_df = pd.read_csv(save_path)
            print(f"Loaded {len(existing_df)} existing results from {save_path}")
        except Exception as e:
            print(f"Error loading existing results: {e}")

    # 그리드 생성
    grid = {}
    
    def process_grid_arg(arg_list):
        return generate_range(float(arg_list[0]), float(arg_list[1]), int(arg_list[2]), arg_list[3])

    # Model Groupings for Grids
    if args.model in ['EASE', 'LAE']:
        grid['reg_p'] = process_grid_arg(args.reg_p_grid)
    
    elif args.model == 'EDLAE':
        grid['drop_p'] = process_grid_arg(args.dropout_grid)
    
    elif args.model == 'RLAE':
        grid['reg_p'] = process_grid_arg(args.reg_p_grid)
        grid['xi'] = process_grid_arg(args.xi_grid)
        
    elif args.model == 'DLAE':
        grid['dropout_p'] = process_grid_arg(args.dropout_grid)

    elif args.model == 'RDLAE':
        grid['drop_p'] = process_grid_arg(args.dropout_grid)
        grid['xi'] = process_grid_arg(args.xi_grid)

    elif args.model in ['DAN_EASE', 'DAN_LAE', 'EASE_DAN']:
        grid['reg_p'] = process_grid_arg(args.reg_p_grid)
        grid['alpha'] = process_grid_arg(args.alpha_grid)
        grid['beta'] = process_grid_arg(args.beta_grid)

    elif args.model == 'DAN_RLAE':
        grid['reg_p'] = process_grid_arg(args.reg_p_grid)
        grid['alpha'] = process_grid_arg(args.alpha_grid)
        grid['beta'] = process_grid_arg(args.beta_grid)
        grid['xi'] = process_grid_arg(args.xi_grid)

    elif args.model == 'DAN_DLAE':
        grid['alpha'] = process_grid_arg(args.alpha_grid)
        grid['beta'] = process_grid_arg(args.beta_grid)
        grid['dropout_p'] = process_grid_arg(args.dropout_grid)

    elif args.model in ['ASPIRE_EASE', 'ASPIRE_LAE', 'CLAE']:
        grid['reg_lambda'] = process_grid_arg(args.reg_lambda_grid)
        grid['alpha'] = process_grid_arg(args.alpha_grid)

    elif args.model == 'ASPIRE_RLAE':
        grid['reg_lambda'] = process_grid_arg(args.reg_lambda_grid)
        grid['alpha'] = process_grid_arg(args.alpha_grid)
        grid['xi'] = process_grid_arg(args.xi_grid)

    elif args.model in ['ASPIRE_DLAE', 'DCLAE']:
        grid['alpha'] = process_grid_arg(args.alpha_grid)
        grid['dropout_p'] = process_grid_arg(args.dropout_grid)

    elif args.model == 'IPS_LAE':
        grid['reg_lambda'] = process_grid_arg(args.reg_lambda_grid)
        grid['wbeta'] = process_grid_arg(args.wbeta_grid)
        grid['wtype'] = [args.wtype]

    elif args.model == 'GFCF':
        grid['alpha'] = process_grid_arg(args.alpha_grid)

    keys = list(grid.keys())
    combinations = [dict(zip(keys, v)) for v in product(*grid.values())]
    
    print(f"Total experiments: {len(combinations)}")
    
    is_strong = (args.mode == 'strong')
    
    for params in tqdm(combinations, desc=f"Searching {args.model}"):
        if is_already_done(existing_df, params):
            continue

        res = run_experiment(args.model, args.dataset, params, args.gpu, is_strong)
        if res:
            res_df = pd.DataFrame([res])
            if os.path.exists(save_path):
                res_df.to_csv(save_path, mode='a', header=False, index=False)
            else:
                res_df.to_csv(save_path, mode='w', header=True, index=False)
            
            if existing_df is None:
                existing_df = res_df
            else:
                existing_df = pd.concat([existing_df, res_df], ignore_index=True)

    print(f"\nFinished! Results saved to {save_path}")

if __name__ == "__main__":
    main()
