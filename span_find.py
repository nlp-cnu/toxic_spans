# This file is used to isolated the toxic spans inside the data for better reading of the data.
"""
@author Jack Lynam
@date 10/04/2020
"""
import numpy as np
import os
import pandas as pd
import csv
from ast import literal_eval
import more_itertools as mit


def extract_toxic_span(span_list, text):
    """
    :param span_list: a list of the indices that indicate a toxic text
    :param text: a text to find toxic spans in
    :return: A list of toxic words/phrases as indicated by span_list
    """
    if len(span_list) == 0:
        return None
    intervals = [list(group) for group in mit.consecutive_groups(span_list)]
    return [text[interval[0]:interval[-1] + 1] for interval in intervals]


def generate_spans(read_file, write_file):
    """
    :param read_file: This is the file to read data from, should be a csv
    :param write_file: This is the file to write new data to, should be a csv
    :return: no return
    """
    trial = pd.read_csv(read_file)
    trial['spans'] = trial.spans.apply(literal_eval)
    trial['toxic'] = [extract_toxic_span(span_list, text) for span_list, text in zip(trial['spans'], trial['text'])]
    trial.to_csv(write_file)


if __name__ == '__main__':
    generate_spans(os.path.join('data', 'tsd_trial.csv'),
                   os.path.join('data', 'tsd_trial_extracted.csv'))
