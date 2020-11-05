# this program will utilize the evaluation script and cross validation script to check the F1 score od the dict_approach
"""
@author rafaeldiaz
@date 11/05/2020
"""
import cross_validation as cv
import dict_approach as da
import system_evaluation as se


def evaluation(eval_path, train_path):
    """
    :param: eval_path - path leading to evaluation file
    :param: train_path - path leading to training file
    :return:
    """

    #


if __name__ == '__main__':
    # first separate file into buckets.
    # it is already separated on my machine from using cross_validation.py
    # with those buckets, use a for loop to iterate through each combination
    train_paths = ["train_1234.csv", "train_1235.csv", "train_1245.csv", "train_1345.csv", "train_2345.csv"]
    for index, i in enumerate(range(1, 6)):
        eval_path = "eval_" + str(i) + ".csv"
        train_path = train_paths[index]

        # first run dictionary_approach to find the indices of the toxic words, it will be returned in a 2d list

        # then, find the actual spans by looking at the eval path

        # feed those two results into the evaluation script

        # eval script needs (text, truth, prediction)

