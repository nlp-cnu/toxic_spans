# this program will utilize the evaluation script and cross validation script to check the F1 score of the dict_approach
"""
@author rafaeldiaz
@date 11/05/2020
"""
import cross_validation as cv
import dict_approach as da
import system_evaluation as se
import pandas as pd
from ast import literal_eval
import numpy as np
import os

if __name__ == '__main__':
    # first separate file into buckets.
    # it is already separated on my machine from using cross_validation.py
    training_data_path = os.path.join("data", "combined_data.csv")
    five_buckets = cv.divide_data(training_data_path)

    # train_1234.csv | eval_5.csv
    cv.create_csv_files(five_buckets[0], five_buckets[1], five_buckets[2], five_buckets[3], five_buckets[4],
                        'train_1234.csv', 'eval_5.csv')

    # train_1235.csv | eval_4.csv
    cv.create_csv_files(five_buckets[0], five_buckets[1], five_buckets[2], five_buckets[4], five_buckets[3],
                        'train_1235.csv', 'eval_4.csv')

    # train_1245.csv | eval_3.csv
    cv.create_csv_files(five_buckets[0], five_buckets[1], five_buckets[3], five_buckets[4], five_buckets[2],
                        'train_1245.csv', 'eval_3.csv')

    # train_1345.csv | eval_2.csv
    cv.create_csv_files(five_buckets[0], five_buckets[2], five_buckets[3], five_buckets[4], five_buckets[1],
                        'train_1345.csv', 'eval_2.csv')

    # train_2345.csv | eval_1.csv
    cv.create_csv_files(five_buckets[1], five_buckets[2], five_buckets[3], five_buckets[4], five_buckets[0],
                        'train_2345.csv', 'eval_1.csv')
    # with those buckets, use a for loop to iterate through each combination
    overall_f1 = []
    train_paths = ["train_1234.csv", "train_1235.csv", "train_1245.csv", "train_1345.csv", "train_2345.csv"]
    for index, i in enumerate(range(1, 6)):

        eval_path = "eval_" + str(i) + ".csv"
        da_path = "da_" + str(i) + ".csv"
        train_path = train_paths[index]

        # first run dictionary_approach to find the indices of the toxic words, it will be returned in a 2d list
        prediction = da.compare_data(eval_path, train_path, da_path)
        # then, find the actual spans by looking at the eval path
        eval = pd.read_csv(eval_path)
        series = eval['spans']
        truth = []
        for item in series:
            truth.append(literal_eval(item))
        # feed those two results into the evaluation script

        # eval script needs (text, truth, prediction)
        list_f1 = []
        for list_index, post in enumerate(eval['text']):
            list_f1.append(se.score_response(post, truth[list_index], prediction[list_index]))
        overall_f1.append(np.mean(list_f1))
        print(
            "For {} as training, {} as evaluation, the F1 score was {}".format(train_path, eval_path, np.mean(list_f1)))
    print("Overall F1 Score {}".format(np.mean(overall_f1)))
