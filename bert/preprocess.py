import os
from ast import literal_eval

import pandas as pd
import numpy as np

from nltk.tokenize import TweetTokenizer


def read_raw_data(read_file):
    """
    Reads the raw data from the train and trial sets
    :param read_file: File path to read from
    :return: pandas df with all the data
    """
    # Code as given by Semeval 2021 task 5 organizers
    df = pd.read_csv(read_file)
    df['spans'] = df.spans.apply(literal_eval)
    return df


def tokenize_text(text):
    """
    Tokenizes text using nltk.word_tokenize
    :param text: the text to word tokenize
    :return: returns a tokenized list of words and punctuation
    """
    # Uses the nltk TweetTokenizer
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens


def space_between_tokens(text, tokens):
    """
    Utility method that finds the space between two tokens
    :param text: The text to be examined
    :param tokens: Tokenized list
    :return: List of spaces between tokens
    """
    # Initializes the array and a counter variable. curr_ind starts at -1 for indexing purposes
    list_of_spaces = []
    curr_ind = -1
    # iterates over the tokens
    for i in range(len(tokens) - 1):
        # Initializes the length of the first token, which is skipped over
        first_tok_len = len(tokens[i])
        # Initializes the first character of the next token
        second_tok_start = tokens[i + 1][0]
        # Skips over the first token
        curr_ind += first_tok_len
        # initializes a counter
        temp_spaces = 0
        # Iterates until the first index of the next token is found
        while text[curr_ind + 1] != second_tok_start:
            curr_ind += 1
            temp_spaces += 1
        # Adds the next number of spaces to the full list
        list_of_spaces.append(temp_spaces)
    return list_of_spaces


def assign_spans(text):
    """
    Creates a tuple that assigns the indices of a word token to that token
    :param text: text to be assigned
    :return: list of tuples that contain a word and the indices of that word
    """
    # Strips any leading or trailing spaces
    text = text.strip()
    # Creates a tokenization of the text
    tokens = tokenize_text(text)
    # Creates a numpy array of all indices in the text
    # Finds all of the spaces between the tokens
    spaces = space_between_tokens(text, tokens)
    # Adds a dummy value to avoid IndexOutOfBoundsError
    spaces.append(0)
    # Initializes index variable and token-span tuple list
    curr_ind = 0
    token_span_list = []
    # Iterates over all the tokens
    for i, token in enumerate(tokens):
        # Checks for a token of length one, if so the index is assigned appropriately
        if len(token) == 1:
            tup = (token, [curr_ind])
        # otherwise the indices are appropriately generated and assigned
        else:
            tup = (token, list(np.arange(curr_ind, curr_ind + len(token))))
        # The index variable is updated and the span list is updated
        curr_ind += len(token) + spaces[i]
        token_span_list.append(tup)
    return token_span_list


def tag_toxic_spans(text, toxic_span):
    """
    Tags all instances of toxic words with BIO
    :param text: Text to be tagged
    :param toxic_span: Predefined toxic characters
    :return: List of tokens and their tag as tuple
    """
    # Grabs the token-span tuple list
    token_span_tuple_list = assign_spans(text)
    token_tags = []
    # Initializes the labels
    b_tox = 'B-Tox'
    i_tox = 'I-Tox'
    out = 'O'
    # Assignes the previous tag to an Out
    prev_tag = out
    # iterates over all the tokens and spans in the list
    for token, token_span in token_span_tuple_list:
        # checks to see if any index of the current token is in the toxic_span list
        check = any(item in token_span for item in toxic_span)
        if check:
            # if the previous tag was an Out, B-Tox is assigned, otherwise I-Tox is assigned
            if prev_tag == out:
                token_tags.append((token, b_tox))
                prev_tag = b_tox
            else:
                token_tags.append((token, i_tox))
                prev_tag = i_tox
        # If the check fails, the token is assigned an Out
        else:
            token_tags.append((token, out))
            prev_tag = out
    return token_tags