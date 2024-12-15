import  os
import  random
from  tqdm import tqdm
import os
import random
from tqdm import tqdm

def split_seq_data(file_path):
    """Split sequence data for recommendation."""
    dst_path = os.path.dirname(file_path)
    train_path = os.path.join(dst_path, "games_seq_train.txt")
    val_path = os.path.join(dst_path, "games_seq_val.txt")
    test_path = os.path.join(dst_path, "games_seq_test.txt")
    meta_path = os.path.join(dst_path, "games_seq_meta.txt")

    users_mapping = {}
    items_mapping = {}
    user_idx, item_idx = 1, 1
    history = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Processing lines"):
            # 去除行尾的换行符并检查是否为空行
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith('#'):
                continue  # 跳过空行或注释行

            try:
                # 使用正确的分隔符 "::" 来分割数据，并确保有足够的值
                parts = stripped_line.split("::")
                if len(parts) != 4:
                    print(f"Warning: Invalid number of fields on line - {stripped_line}")
                    continue  # 跳过字段数量不正确的行

                user, item, score, timestamp = parts
            except ValueError as e:
                print(f"Error processing line - {stripped_line}: {e}")
                continue  # 跳过分隔符错误的行

            # 添加用户到映射表中
            if user not in users_mapping:
                users_mapping[user] = str(user_idx)
                user_idx += 1

            # 添加物品到映射表中
            if item not in items_mapping:
                items_mapping[item] = str(item_idx)
                item_idx += 1

            # 更新历史记录
            history.setdefault(users_mapping[user], []).append([items_mapping[item], timestamp])

    # 写入训练集、验证集和测试集
    with open(train_path, 'w', encoding='utf-8') as f1, \
         open(val_path, 'w', encoding='utf-8') as f2, \
         open(test_path, 'w', encoding='utf-8') as f3:
        for user_id in users_mapping.values():
            hist_u = history.get(user_id, [])
            if len(hist_u) < 4:
                continue

            hist_u.sort(key=lambda x: x[1])
            hist = [x[0] for x in hist_u]
            time = [x[1] for x in hist_u]

            f1.write(f"{user_id}\t{' '.join(hist[:-2])}\t{' '.join(time[:-2])}\n")
            f2.write(f"{user_id}\t{' '.join(hist[:-2])}\t{' '.join(time[:-2])}\t{hist[-2]}\n")
            f3.write(f"{user_id}\t{' '.join(hist[:-1])}\t{' '.join(time[:-1])}\t{hist[-1]}\n")

    # 写入元数据文件
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(f"{len(users_mapping)}\t{len(items_mapping)}")

    return train_path, val_path, test_path, meta_path
