# this program will convert our data to BERT's expected input
"""
@author rafaeldiaz
@date 11/12/2020
"""
import os
import pandas as pd


def read_spans(read_file):
    """

    :param read_file:
    :return: not sure yet
    """

    read = pd.read_csv(read_file)

    print(read)


if __name__ == "__main__":
    path = os.path.join('data', 'combined_data.csv')
    read_spans(path)
    print("Placeholder")

    # first find spans in text

    # second, tokenize each word in a line,

    # third, write to a file each word and their respective token
    # note, for now, we may want to limit how many lines we do as a test basis and then figure out how to organize the
    # complete file from there.
