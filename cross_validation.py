import os.path
import csv
import pandas as pd
import math


def divide_data(file_path):

    # read file and shuffle rows
    lines = pd.read_csv(file_path, header=0)
    shuffled = lines.sample(frac=1).reset_index(drop=True)

    # create 5 buckets, each containing roughly 20% of data
    num_rows = len(shuffled.index)
    bucket_size = math.ceil(num_rows / 5)

    buckets = []
    for i in range(5):
        start_index = (bucket_size * i)
        end_index = bucket_size * (i+1)
        buckets.append((shuffled.iloc[start_index:end_index]).reset_index(drop=True))

    for bucket in buckets:
        print(bucket)

    return buckets


def create_csv_files(bucket_one, bucket_two, bucket_three, bucket_four, eval_bucket, train_file_name, eval_file_name):
    data_frame = pd.concat([bucket_one, bucket_two, bucket_three, bucket_four], ignore_index=True)
    data_frame.to_csv(train_file_name, index=False)
    eval_bucket.to_csv(eval_file_name)


if __name__ == "__main__":
    training_data_path = "tsd_train.csv"
    five_buckets = divide_data(training_data_path)

    # train_1234.csv | eval_5.csv
    create_csv_files(five_buckets[0], five_buckets[1], five_buckets[2], five_buckets[3], five_buckets[4],
                     'train_1234.csv', 'eval_5.csv')

    # train_1235.csv | eval_4.csv
    create_csv_files(five_buckets[0], five_buckets[1], five_buckets[2], five_buckets[4], five_buckets[3],
                     'train_1235.csv', 'eval_4.csv')

    # train_1245.csv | eval_3.csv
    create_csv_files(five_buckets[0], five_buckets[1], five_buckets[3], five_buckets[4], five_buckets[2],
                     'train_1245.csv', 'eval_3.csv')

    # train_1345.csv | eval_2.csv
    create_csv_files(five_buckets[0], five_buckets[2], five_buckets[3], five_buckets[4], five_buckets[1],
                     'train_1345.csv', 'eval_2.csv')

    # train_2345.csv | eval_1.csv
    create_csv_files(five_buckets[1], five_buckets[2], five_buckets[3], five_buckets[4], five_buckets[0],
                     'train_2345.csv', 'eval_1.csv')
