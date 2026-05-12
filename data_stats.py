import os
import csv
import numpy as np
from collections import defaultdict

def analyze_dataset(ds_path):
    files = ['train.txt', 'valid_in.txt', 'valid.txt', 'test_in.txt', 'test.txt']
    
    user_interactions = defaultdict(int)
    item_interactions = defaultdict(int)
    split_counts = {f: 0 for f in files}
    
    all_users = set()
    all_items = set()
    
    for f_name in files:
        f_path = os.path.join(ds_path, f_name)
        if not os.path.exists(f_path):
            continue
            
        with open(f_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                uid = parts[0]
                items = parts[1:]
                
                num_items = len(items)
                split_counts[f_name] += num_items
                user_interactions[uid] += num_items
                all_users.add(uid)
                for iid in items:
                    item_interactions[iid] += 1
                    all_items.add(iid)

    n_users = len(all_users)
    n_items = len(all_items)
    total_inter = sum(split_counts.values())
    
    if n_users == 0 or n_items == 0:
        return None

    # 계산된 지표들
    sparsity = (1 - (total_inter / (n_users * n_items))) * 100
    u_counts = list(user_interactions.values())
    i_counts = list(item_interactions.values())

    # User-Item degree correlation calculation
    u_degs = np.zeros(total_inter, dtype=int)
    i_degs = np.zeros(total_inter, dtype=int)
    idx = 0
    for f_name in files:
        f_path = os.path.join(ds_path, f_name)
        if not os.path.exists(f_path):
            continue
        with open(f_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                uid = parts[0]
                items = parts[1:]
                d_u = user_interactions[uid]
                for iid in items:
                    u_degs[idx] = d_u
                    i_degs[idx] = item_interactions[iid]
                    idx += 1
    
    degree_corr = np.corrcoef(u_degs, i_degs)[0, 1]
    
    return {
        'Users': n_users,
        'Items': n_items,
        'Interactions': total_inter,
        'Sparsity (%)': f"{sparsity:.4f}",
        'Avg Inter/User': f"{np.mean(u_counts):.2f}",
        'Max Inter/User': np.max(u_counts),
        'Avg Inter/Item': f"{np.mean(i_counts):.2f}",
        'Degree Corr': f"{degree_corr:.4f}",
        'Train': split_counts['train.txt'],
        'Valid': split_counts['valid_in.txt'] + split_counts['valid.txt'],
        'Test': split_counts['test_in.txt'] + split_counts['test.txt']
    }

def main():
    base_path = 'strong/data'
    datasets = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    
    results = []
    for ds in datasets:
        print(f"Analyzing {ds}...")
        stats = analyze_dataset(os.path.join(base_path, ds))
        if stats:
            stats['Dataset'] = ds
            results.append(stats)

    # 컬럼 순서 조정
    cols = ['Dataset', 'Users', 'Items', 'Interactions', 'Sparsity (%)', 
            'Avg Inter/User', 'Max Inter/User', 'Avg Inter/Item', 'Degree Corr',
            'Train', 'Valid', 'Test']
    
    output_file = 'data_stats.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nDetailed statistics saved to {output_file}")

if __name__ == "__main__":
    main()
