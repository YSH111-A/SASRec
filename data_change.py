import numpy as np


def prepare_for_keras(data, seq_len, neg_num):
    """Prepare data for Keras model input without changing the original structure."""
    prepared_data = []

    for entry in data:
        click_seq = entry['click_seq']
        pos_item = entry['pos_item']
        neg_items = entry['neg_items']

        # 确保 'click_seq' 长度固定
        if len(click_seq) >= seq_len:
            click_seq = click_seq[-seq_len:]
        else:
            click_seq = [0] * (seq_len - len(click_seq)) + click_seq

        # 确保 'neg_items' 长度固定
        if len(neg_items) < neg_num:
            neg_items = neg_items + [0] * (neg_num - len(neg_items))
        elif len(neg_items) > neg_num:
            neg_items = neg_items[:neg_num]

        prepared_entry = {
            'click_seq': np.array(click_seq),
            'pos_item': np.array(pos_item),
            'neg_items': np.array(neg_items)
        }
        prepared_data.append(prepared_entry)

    return prepared_data


