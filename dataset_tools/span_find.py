# This file is used to isolated the toxic spans inside the data for better reading of the data.
"""
@author Jack Lynam
@date 10/04/2020
"""
import numpy as np
import os
import pandas as pd
import transformers
from ast import literal_eval
import more_itertools as mit
import convert_to_bert as ctb


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

def max_text_length(read_file):
    trial = pd.read_csv(read_file)
    return max(trial['text'])

def token_stats(read_file):
    """
    Finds meaningful statistics on the length of the token lists
    :param read_file: File to read from
    """
    df = ctb.read_raw_data(read_file)
    texts = df['text']
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    sequence_lengths = [len(tokenizer.encode(text)) for text in texts]
    print('Max sequence found:', max(sequence_lengths))
    print('Mean sequence length:', np.mean(sequence_lengths))
    print('Std Deviation of sequence length:', np.std(sequence_lengths))


if __name__ == '__main__':
    generate_spans(os.path.join('data', 'tsd_trial.csv'),
                   os.path.join('data', 'tsd_trial_extracted.csv'))
