# This file will be used for running the evaluation script
"""
@author Jack Lynam
@date 10/11/2020
"""

import numpy as np

post = '"Wythe thinks Republican members of the House should settle their differences in caucus meetings — which are private."  That statement should scare the h*ll out of  any voter.  Another advocate for a state government functioning  behind closed doors is exactly who we do not need in the Alaska Legislature.  The  system she apparently adores (the secretive standing caucus system)has just about pounded this state into the ground.  Also, does she understand caucus membership is all about crushing any minority view and has little to do with party platforms and everything to do with the quid pro quo of trading toady behavior for spending in deals that are done behind closed doors?  Stupid is as stupid does..........'
ground_truth = [685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708]
prediction = [21, 34, 82, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708]


def score_response(post, ground_truth, prediction):
    """
    Evaluation for a single response
    :param ground_truth: ground truth for a post
    :param prediction: system prediction for a post
    :return: F1 score for a single truth/prediction pair
    """

    # If no gold spans are given, the prediction must be an empty list for points to be awarded
    if len(ground_truth) == 0 and len(prediction) == 0:
        return 1
    elif (len(ground_truth) == 0 and len(prediction) != 0) or (len(ground_truth) != 0 and len(prediction) == 0):
        return 0
    else:
        # masking the post with booleans
        # If an index of post is found in the ground truth, True is assigned
        gt_mask = np.zeros(len(post), dtype=bool)
        gt_mask[ground_truth] = True
        # If an index of post is found in the prediction, True is assigned
        pred_mask = np.zeros(len(post), dtype=bool)
        pred_mask[prediction] = True
        # True_pos is the number of correctly assigned Trues in prediction
        agree = [p for g, p in zip(gt_mask, pred_mask) if g == p]
        disagree = [p for g, p in zip(gt_mask, pred_mask) if g != p]
        true_pos = sum(agree)
        # true_pos is the number of correctly assigned Trues in prediction
        # If true_pos is 0, the F1 score is automatically 0
        if true_pos == 0:
            return 0
        false_pos = sum(disagree)
        false_neg = len(disagree) - false_pos
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        return (2 * precision * recall) / (precision + recall)


print(score_response(post, ground_truth, prediction))
