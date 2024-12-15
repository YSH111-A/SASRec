import numpy as np
import tensorflow as tf
from sklearn.metrics import label_ranking_average_precision_score


def Evaluate_model(model, test_data, k=10):
    """Evaluate the model on the test data and return HR, MRR, and NDCG scores.

    Args:
        model (tf.keras.Model): The trained SASRec model.
        test_data (dict): A dictionary containing 'click_seq', 'pos_item', and 'neg_items'.
        k (int): The number of top items to consider for evaluation.

    Returns:
        dict: A dictionary containing HR, MRR, and NDCG scores.
    """

    def hit_rate(rank_list, target_item):
        return int(target_item in rank_list)

    def reciprocal_rank(rank_list, target_item):
        if target_item in rank_list:
            return 1 / (rank_list.index(target_item) + 1)
        else:
            return 0.0

    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        return 0.0

    def ndcg_at_k(r, k):
        idcg = dcg_at_k(sorted(r, reverse=True), k)
        if not idcg:
            return 0.0
        return dcg_at_k(r, k) / idcg

    hr_list, mrr_list, ndcg_list = [], [], []

    # Iterate over each user in the test data
    for user_data in test_data:
        click_seq = user_data['click_seq']
        pos_item = user_data['pos_item']
        neg_items = user_data['neg_items']

        # Predict scores for positive and negative items
        input_data = {
            'click_seq': np.array([click_seq]),
            'pos_item': np.array([pos_item]),
            'neg_item': np.array([neg_items])
        }
        predictions = model.predict(input_data, batch_size=1).flatten()

        # Get the score for the positive item
        pos_score = predictions[0]

        # Get the scores for negative items
        neg_scores = predictions[1:]

        # Combine positive and negative items and their scores
        item_scores = list(zip([pos_item] + list(neg_items), [pos_score] + list(neg_scores)))

        # Sort by score in descending order
        item_scores.sort(key=lambda x: x[1], reverse=True)

        # Extract the top-k items
        top_k_items = [item for item, score in item_scores[:k]]
        top_k_scores = [score for item, score in item_scores[:k]]

        # Calculate HR, MRR, and NDCG
        hr_list.append(hit_rate(top_k_items, pos_item))
        mrr_list.append(reciprocal_rank(top_k_items, pos_item))
        ndcg_list.append(ndcg_at_k([1] + [0] * len(neg_scores), k))

    # Calculate average metrics
    eval_dict = {
        'hr': np.mean(hr_list),
        'mrr': np.mean(mrr_list),
        'ndcg': np.mean(ndcg_list)
    }

    return eval_dict