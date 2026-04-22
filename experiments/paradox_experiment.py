import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import gc

# 1. Path setup
# 이 스크립트의 위치(experiments/)를 기준으로 프로젝트 루트와 코드 경로를 계산합니다.
EXPERIMENT_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.abspath(os.path.join(EXPERIMENT_PATH, ".."))
CODE_PATH = os.path.join(ROOT_PATH, 'strong', 'code')
sys.path.append(CODE_PATH)

# 2. Mocking world.args before importing other modules
import argparse
def mock_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='abook')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--testbatch', type=int, default=2048)
    args, unknown = parser.parse_known_args()
    
    config = argparse.Namespace()
    config.dataset = args.dataset
    config.model = 'ASPIRE_EASE'
    config.seed = 2020
    config.gpu = args.gpu
    config.testbatch = args.testbatch
    config.topks = "[20, 50, 100]"
    config.reg_p = 10.0
    config.reg_lambda = 10.0
    config.alpha = 0.5
    config.xi = 0.0
    config.diag_const = True
    config.drop_p = 0.0
    config.dropout_p = 0.0
    config.wbeta = 0.0
    config.wtype = 'logsigmoid'
    config.multicore = 0
    return config

# Monkey patch parse_args in parse.py
import parse
parse.parse_args = mock_parse_args

# 3. Import modules with CWD change to handle relative paths in register.py
# register.py가 "../data/" 경로를 사용하므로, 임포트 시점에만 위치를 변경합니다.
original_cwd = os.getcwd()
os.chdir(CODE_PATH)

import world
import utils
import register
from register import dataset
import model

# Initialize pscore as in main.py to avoid AttributeError in Procedure.py
item_freq = np.array(dataset.UserItemNet.sum(axis=0)).squeeze()
world.pscore = np.maximum((item_freq / item_freq.max()) ** 0.5, 10e-3)

# 다시 원래 작업 디렉토리로 복구
os.chdir(original_cwd)

def evaluate_group(Recmodel, users, dataset, k=100):
    Recmodel.eval()
    testDict = dataset.testDict
    group_users = [u for u in users if u in testDict]
    if not group_users:
        return 0.0
    
    u_batch_size = world.config['test_u_batch_size']
    all_ndcg = []
    
    with torch.no_grad():
        for batch_users in utils.minibatch(group_users, batch_size=u_batch_size):
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long().to(world.device)
            rating = Recmodel.getUsersRating(batch_users_gpu)
            allPos = dataset.getTestUserPosItems(batch_users)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=k)
            rating_K = rating_K.cpu().numpy()
            r = utils.getLabel(groundTrue, rating_K)
            batch_ndcg_sum = utils.NDCGatK_r(groundTrue, r, k)
            all_ndcg.append(batch_ndcg_sum)
            
    return sum(all_ndcg) / len(group_users)

def run_paradox_experiment():
    print(f"Starting Paradox of Correction Experiment on {world.dataset}")
    
    user_degrees = dataset.users_D
    test_users = np.array(list(dataset.testDict.keys()))
    
    # Filtering: Degree가 1인 유저는 보정 효과가 없으므로 제외 (손맛을 위해)
    mask = user_degrees[test_users] > 1
    valid_test_users = test_users[mask]
    valid_degrees = user_degrees[valid_test_users]
    
    sorted_indices = np.argsort(valid_degrees)[::-1]
    n_total = len(valid_test_users)
    
    # 대비를 위해 더 극단적인 그룹핑 (Top 10% vs Bottom 10%)
    cutoff = int(n_total * 0.1)
    
    head_users = valid_test_users[sorted_indices[:cutoff]]
    tail_users = valid_test_users[sorted_indices[-cutoff:]]
    mid_users  = valid_test_users[sorted_indices[cutoff:-cutoff]]
    
    print(f"Filtered Users (>1 interaction): {n_total}")
    print(f"Head Users (Top 10%): {len(head_users)} (Avg Degree: {np.mean(user_degrees[head_users]):.2f})")
    print(f"Tail Users (Bottom 10%): {len(tail_users)} (Avg Degree: {np.mean(user_degrees[tail_users]):.2f})")
    
    gammas = np.linspace(0.0, 2.0, 21)
    results = []
    world.config['reg_lambda'] = 10.0 
    
    for gamma in gammas:
        print(f"\nEvaluating Gamma (alpha) = {gamma:.2f}")
        world.config['alpha'] = gamma
        rec_model = model.ASPIRE_EASE(world.config, dataset)
        
        eps = 1e-12
        head_ndcg = evaluate_group(rec_model, head_users, dataset)
        mid_ndcg  = evaluate_group(rec_model, mid_users, dataset)
        tail_ndcg = evaluate_group(rec_model, tail_users, dataset)
        
        # Weight Std (Tail is the main concern)
        tail_weights = 1.0 / (np.power(user_degrees[tail_users], gamma) + eps)
        tail_std = np.std(tail_weights)
        
        print(f"  NDCG@100 -> Head: {head_ndcg:.4f} | Mid: {mid_ndcg:.4f} | Tail: {tail_ndcg:.4f}")
        
        results.append({
            'gamma': gamma, 
            'head_ndcg': head_ndcg, 'mid_ndcg': mid_ndcg, 'tail_ndcg': tail_ndcg,
            'tail_std': tail_std
        })
        del rec_model
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    os.makedirs(os.path.join(ROOT_PATH, 'exp_result'), exist_ok=True)
    csv_path = os.path.join(ROOT_PATH, 'exp_result', f'paradox_experiment_{world.dataset}.csv')
    df.to_csv(csv_path, index=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Performance Comparison
    ax1.plot(df['gamma'], df['head_ndcg'], marker='o', label='Head (Top 20%)', color='blue')
    ax1.plot(df['gamma'], df['mid_ndcg'],  marker='D', label='Mid (Middle 60%)', color='green')
    ax1.plot(df['gamma'], df['tail_ndcg'], marker='s', label='Tail (Bottom 20%)', color='red')
    ax1.set_xlabel('Correction Strength ($\gamma$)', fontsize=12)
    ax1.set_ylabel('Performance (NDCG@100)', fontsize=12)
    ax1.set_title(f'Paradox of Correction: Performance ({world.dataset})', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7); ax1.legend()
    
    # Plot 2: Tail Weight Instability
    ax2.plot(df['gamma'], df['tail_std'], marker='s', label='Tail Weight Std', color='red')
    ax2.set_yscale('log')
    ax2.set_xlabel('Correction Strength ($\gamma$)', fontsize=12)
    ax2.set_ylabel('Weight Standard Deviation (log scale)', fontsize=12)
    ax2.set_title('Why Tail Fails: Weight Instability', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7); ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(ROOT_PATH, 'exp_result', f'paradox_experiment_{world.dataset}.png')
    plt.savefig(plot_path)
    print(f"\nResults and plots saved in 'exp_result/'")
    ax2.grid(True, linestyle='--', alpha=0.7); ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(ROOT_PATH, 'exp_result', f'paradox_experiment_{world.dataset}.png')
    plt.savefig(plot_path)
    print(f"\nResults and plots saved in 'exp_result/'")

if __name__ == "__main__":
    run_paradox_experiment()
