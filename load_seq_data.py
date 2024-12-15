import numpy as np

def parse_line(line):
    """Parse a single line of the dataset into user_id and item sequence, ignoring ratings."""
    parts = line.strip().split('\t')
    if len(parts) < 3:
        return None, None
    try:
        user_id = int(parts[0])
        items = [int(item) for item in parts[1].split()]
        return user_id, items
    except ValueError:
        return None, None

def load_train_data(file_path, seq_len=50, neg_num=4, max_item_num=3706, contain_user=False):
    """Load and preprocess training data from a file.

    Args:
        file_path (str): Path to the dataset file.
        seq_len (int): Length of the input sequence.
        neg_num (int): Number of negative samples for each positive sample.
        max_item_num (int): The largest item index + 1.
        contain_user (bool): Whether the dataset contains user information.

    Returns:
        list: A list of dictionaries containing 'click_seq', 'pos_item', and 'neg_items'.
    """
    data = []

    with open(file_path, 'r') as f:
        for line in f:
            user_id, items = parse_line(line)
            if user_id is None or items is None or len(items) < seq_len + 2:
                continue

            train_items = items[:-2]
            pos_item = items[-2]

            # Pad or truncate the sequence to the specified length
            click_seq = train_items[-seq_len:] if len(train_items) >= seq_len else ([0] * (seq_len - len(train_items)) + train_items)

            # Generate negative samples
            neg_samples = set()
            while len(neg_samples) < neg_num:
                neg_sample = np.random.randint(1, max_item_num)
                if neg_sample not in items:
                    neg_samples.add(neg_sample)
            neg_samples = list(neg_samples)

            if contain_user:
                data.append({'click_seq': click_seq, 'pos_item': pos_item, 'neg_items': neg_samples, 'user_id': user_id})
            else:
                data.append({'click_seq': click_seq, 'pos_item': pos_item, 'neg_items': neg_samples})

    return data

