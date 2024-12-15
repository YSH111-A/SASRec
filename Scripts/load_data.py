import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from collections import defaultdict


def load_data(file_path, neg_num, max_item_num):
    """L" " "加载movielens数据集并生成负样本。

        参数:
        :param file_path:一个字符串。文件路径。
        :param neg_num:一个标量(int)。每个阳性样本的阴性样本数。
        :param max_item_num:一个标量(int)。项目的最大索引。
        返回:
        :return:包含用户、肯定项和否定项的字
    """
    # 读取数据
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            user, item, rating, timestamp = line.strip().split('::')
            data.append((int(user), int(item)))

    # 构建用户-物品交互字典
    user_items_dict = defaultdict(set)
    for user, item in data:
        user_items_dict[user].add(item)

    # 打乱数据顺序
    np.random.shuffle(data)

    # 生成负样本
    neg_items = []
    for user, pos_item in tqdm(data, desc="Generating negative samples"):
        neg_item_set = set()
        while len(neg_item_set) < neg_num:
            neg_item = random.randint(1, max_item_num)
            if neg_item not in user_items_dict[user]:
                neg_item_set.add(neg_item)
        neg_items.append(list(neg_item_set))

    # 返回结果作为字典
    return {
        'user': np.array([d[0] for d in data], dtype=int),
        'pos_item': np.array([d[1] for d in data], dtype=int),
        'neg_item': np.array(neg_items, dtype=object)
    }
