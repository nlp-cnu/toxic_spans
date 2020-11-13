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



if __name__ == "__main__":
    print("Placeholder")

    # first we want to read our given file and find the toxic words from the text.

    # then, with that knowledge, we want to read the text, and if the word is considered toxic via spans,
    # we should assign it with a toxic token, otherwise assign with a 0 

