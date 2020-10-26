# This file will be used for running the evaluation script
"""
@author Jack Lynam
@date 10/11/2020
"""

import numpy as np

post = '"Wythe thinks Republican members of the House should settle their differences in caucus meetings — which are private."  That statement should scare the h*ll out of  any voter.  Another advocate for a state government functioning  behind closed doors is exactly who we do not need in the Alaska Legislature.  The  system she apparently adores (the secretive standing caucus system)has just about pounded this state into the ground.  Also, does she understand caucus membership is all about crushing any minority view and has little to do with party platforms and everything to do with the quid pro quo of trading toady behavior for spending in deals that are done behind closed doors?  Stupid is as stupid does..........'
ground_truth = [685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708]
prediction = [0, 701, 702, 703, 704, 705, 706, 707, 708]


def score_response(post, ground_truth, prediction):
    """
    Evaluation for a single response
    :param ground_truth: ground truth for a post
    :param prediction: system prediction for a post
    :return: F1 score for a single truth/prediction pair
    """

    # If the ground truth set is empty, the prediction set must be empty, system score is 1
    if len(ground_truth) == 0 and len(prediction) == 0:
        return 1
    # If a system passes a non-empty prediction set when the ground truth set is empty, system score is 0
    # If a system passes an empty prediction set when there is a non-empty ground truth set, system score is 0
    elif (len(ground_truth) == 0 and len(prediction) != 0) or (len(ground_truth) != 0 and len(prediction) == 0):
        return 0
    else:
        # Making a boolean mask of the ground truth set over the length the post
        gt_mask = np.zeros(len(post), dtype=bool)
        gt_mask[ground_truth] = True
        # Making a boolean mask of the prediction set over the length the post
        pred_mask = np.zeros(len(post), dtype=bool)
        pred_mask[prediction] = True
        # The agreement between the ground truth and prediction masks contain information on both
        # the true positives and true negatives.
        # For the agreement, it does not matter which value we keep, both are the same
        agree = [p for g, p in zip(gt_mask, pred_mask) if g == p]
        # The disagreement between the ground truth and prediction masks contain information on both
        # the false positives and false negatives.
        # For the disagreement, we keep the values in the prediction mask because that's what we aim to score
        disagree = [p for g, p in zip(gt_mask, pred_mask) if g != p]
        # true_pos is the number of correctly assigned True in prediction
        true_pos = sum(agree)
        # If true_pos is 0, the F1 score is automatically 0
        if true_pos == 0:
            return 0
        # false_pos is the number of True in the disagreement mask
        false_pos = sum(disagree)
        # false_neg is the number of False in the disagreement mask
        false_neg = len(disagree) - false_pos
        # Precision is calculated
        precision = true_pos / (true_pos + false_pos)
        # Recall is calculated
        recall = true_pos / (true_pos + false_neg)
        # The F1 score is calculated and returned
        return (2 * precision * recall) / (precision + recall)


print('F1 score is:', score_response(post, ground_truth, prediction))
