import subprocess
import os
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import re
import argparse

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
    script_path = os.path.join(base_dir, "code", "main.py")
    
    cmd = [
        "python", script_path,
        "--model", model,
        "--dataset", dataset,
        "--gpu", str(gpu)
    ]
    
    for k, v in params.items():
        cmd.extend([f"--{k}", f"{v:.6f}" if isinstance(v, float) else str(v)])
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
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

def main():
    parser = argparse.ArgumentParser(description="Grid Search for CLAE/DCLAE")
    parser.add_argument('--model', type=str, default='CLAE', choices=['CLAE', 'DCLAE', 'EASE', 'GFCF', 'RLAE', 'RDLAE'])
    parser.add_argument('--dataset', type=str, default='yelp2018')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default='strong', choices=['strong', 'weak'])
    
    # 그리드 설정 인자들
    # 형식: --param start end num scale
    parser.add_argument('--reg_lambda_grid', nargs=4, metavar=('START', 'END', 'NUM', 'SCALE'), 
                        default=[0.1, 100.0, 5, 'log'], help='Grid for reg_lambda')
    parser.add_argument('--alpha_grid', nargs=4, metavar=('START', 'END', 'NUM', 'SCALE'), 
                        default=[0.0, 1.0, 3, 'linear'], help='Grid for alpha')
    parser.add_argument('--beta_grid', nargs=4, metavar=('START', 'END', 'NUM', 'SCALE'), 
                        default=[0.0, 1.0, 3, 'linear'], help='Grid for beta')
    parser.add_argument('--dropout_grid', nargs=4, metavar=('START', 'END', 'NUM', 'SCALE'), 
                        default=[0.1, 0.7, 4, 'linear'], help='Grid for dropout_p (DCLAE only)')
    
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    
    # 그리드 생성
    grid = {}
    
    def process_grid_arg(arg_list):
        return generate_range(float(arg_list[0]), float(arg_list[1]), int(arg_list[2]), arg_list[3])

    if args.model in ['CLAE', 'DCLAE']:
        grid['reg_lambda'] = process_grid_arg(args.reg_lambda_grid)
        grid['alpha'] = process_grid_arg(args.alpha_grid)
        grid['beta'] = process_grid_arg(args.beta_grid)
        if args.model == 'DCLAE':
            grid['dropout_p'] = process_grid_arg(args.dropout_grid)
    
    # 다른 모델 예시 (EASE 등)
    elif args.model == 'EASE':
        grid['reg_p'] = process_grid_arg([10, 1000, 5, 'log'])

    keys = list(grid.keys())
    combinations = [dict(zip(keys, v)) for v in product(*grid.values())]
    
    print(f"Total experiments: {len(combinations)}")
    all_results = []
    
    is_strong = (args.mode == 'strong')
    
    for params in tqdm(combinations, desc=f"Searching {args.model}"):
        res = run_experiment(args.model, args.dataset, params, args.gpu, is_strong)
        if res:
            all_results.append(res)
            df = pd.DataFrame(all_results)
            save_path = f"results/grid_search_{args.model}_{args.dataset}_{args.mode}.csv"
            df.to_csv(save_path, index=False)

    print(f"\nFinished! Results saved to results/grid_search_{args.model}_{args.dataset}_{args.mode}.csv")

if __name__ == "__main__":
    main()
