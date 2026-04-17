import os
import random
from collections import defaultdict

def preprocess_steam_5core(file_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 초기 데이터 로드
    print(f"Reading {file_path}...")
    interactions = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) < 2: continue
            uid, iid = parts[0], parts[1]
            interactions.append((uid, iid))

    # 2. Iterative 5-core Filtering
    print("Applying 5-core filtering...")
    while True:
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)
        for uid, iid in interactions:
            user_counts[uid] += 1
            item_counts[iid] += 1
        
        # 5개 미만인 유저/아이템 식별
        bad_users = {uid for uid, count in user_counts.items() if count < 5}
        bad_items = {iid for iid, count in item_counts.items() if count < 5}
        
        if not bad_users and not bad_items:
            break
            
        # 필터링 진행
        new_interactions = [
            (uid, iid) for uid, iid in interactions 
            if uid not in bad_users and iid not in bad_items
        ]
        
        print(f"  - Filtered: Users={len(bad_users)}, Items={len(bad_items)}. Remaining interactions: {len(new_interactions)}")
        interactions = new_interactions

    # 3. 데이터 구조화 및 ID 매핑
    user_items = defaultdict(list)
    unique_items = set()
    for uid, iid in interactions:
        user_items[uid].append(iid)
        unique_items.add(iid)
        
    users = list(user_items.keys())
    random.shuffle(users)
    item_to_id = {iid: i for i, iid in enumerate(sorted(list(unique_items)))}
    
    print(f"Final Stats: {len(users)} users, {len(unique_items)} items, {len(interactions)} interactions")

    # 4. 분할 (8:1:1)
    num_users = len(users)
    train_end = int(num_users * 0.8)
    valid_end = int(num_users * 0.9)
    
    train_users = users[:train_end]
    valid_users = users[train_end:valid_end]
    test_users = users[valid_end:]

    def save_to_txt(u_list, out_name, is_split=False, in_name=None):
        f_out = open(os.path.join(output_dir, out_name), 'w')
        f_in = open(os.path.join(output_dir, in_name), 'w') if is_split else None
        
        for mapped_uid, uid in enumerate(u_list):
            items = [item_to_id[iid] for iid in user_items[uid]]
            random.shuffle(items)
            
            if is_split:
                split_idx = int(len(items) * 0.8)
                in_items = items[:split_idx]
                target_items = items[split_idx:]
                f_in.write(f"{mapped_uid} {' '.join(map(str, in_items))}\n")
                f_out.write(f"{mapped_uid} {' '.join(map(str, target_items))}\n")
            else:
                f_out.write(f"{mapped_uid} {' '.join(map(str, items))}\n")
        
        f_out.close()
        if f_in: f_in.close()

    print("Saving files...")
    save_to_txt(train_users, 'train.txt')
    save_to_txt(valid_users, 'valid.txt', is_split=True, in_name='valid_in.txt')
    save_to_txt(test_users, 'test.txt', is_split=True, in_name='test_in.txt')

    print(f"Preprocessing complete. Files saved in {output_dir}")

if __name__ == "__main__":
    input_file = '/Users/leejongmin/code/CLAE/strong/data/steam/ratings.dat'
    output_directory = '/Users/leejongmin/code/CLAE/strong/data/steam'
    preprocess_steam_5core(input_file, output_directory)
