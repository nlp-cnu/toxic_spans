# this file will group up only the toxic words and allow for manipulation of just the words
"""
@author Rafael Diaz
@date 10/13/2020
"""
import os
from src import span_find as sf
import matplotlib.pyplot as plt
import pandas as pd
from ast import literal_eval


def create_toxic_list(read_file):
    """
    returns a list of only the words that were considered toxic
    :param read_file:
    :return cleaned_list: a list of words from the spans given a text file
    """
    toxic_list = []
    cleaned_list = []
    data = pd.read_csv(read_file)
    data['spans'] = data.spans.apply(literal_eval)
    data['toxic'] = [sf.extract_toxic_span(span_list, text) for span_list, text in zip(data['spans'], data['text'])]

    for i in data['toxic']:
        if i is not None:
            toxic_list.append(i)

    # just making the list neater by making it 1 dimensional
    for i in toxic_list:
        for j in i:
            cleaned_list.append(j)
    return cleaned_list


def quick_sort(list):
    """

    :param list: list can be anything, will all be sorted
    :return: the list organized alphabetically. additionally, every word inside the list is made to all lower case
    """
    lowered_list = []
    for i in list:
        lowered_list.append(i.lower())
    lowered_list.sort()  # sorting alphabetically allows me to pop the array to get rid of multiple instances
    return lowered_list


def words_and_occurrences(list):
    """

    :param list: list of words that are toxic
    :return: dictionary of words and how often they appear
    """
    organized_list = quick_sort(list)
    word_count = {}
    write_file = os.path.join("data", "words_and_occurrences")
    unique_list = set(organized_list)

    for word in unique_list:
        word_count[word] = organized_list.count(word)

    sort_dict = sorted(word_count.items(), key=lambda x: x[1], reverse=True)  # creates a descending list, with tuples
    # inside. tuples being filled with word in the first value and their count in the second.

    with open(write_file, "w") as file:  # writing into a file the dictionary for an easier visualization of occurrences
        for words, count in sort_dict:
            file.write(words + " " + str(count) + "\n")

    return word_count


def words_and_occurrences_modified(list):
    """
    this one has been modified to only show words that show up at least more than one time
    :param list: list of words that are toxic
    :return: dictionary of words and how often they appear
    """
    organized_list = quick_sort(list)
    word_count = {}
    write_file = os.path.join("data", "words_and_occurrences_modified")
    unique_list = set(organized_list)
    final_dict = {}

    for word in unique_list:
        word_count[word] = organized_list.count(word)

    sort_dict = sorted(word_count.items(), key=lambda x: x[1], reverse=True)  # creates a descending list, with tuples
    # inside. tuples being filled with word in the first value and their count in the second.

    with open(write_file, "w") as file:  # writing into a file the dictionary for an easier visualization of occurrences
        for words, count in sort_dict:
            if count > 1:
                file.write(words + " " + str(count) + "\n")
                final_dict[words] = count

    return final_dict


def create_histogram(dict):
    """
    creates a histogram from a given dictionary
    :param dict: dictionary of toxic words and their occurrences
    """
    x_values = dict.keys()

    plt.hist(x_values, bins=len(dict), range=(0, 200), label=None)
    plt.show()

    # this is not yet done, still a work in progress


if __name__ == '__main__':
    file_path = os.path.join("data", "combined_data_extracted.csv")
    x = create_toxic_list(file_path)
    dictionary_of_words = words_and_occurrences(x)
    greater_dictionary = words_and_occurrences_modified(x)
    create_histogram(greater_dictionary)
