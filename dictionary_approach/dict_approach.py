# this file will begin the dictionary approach for the datasets. 80% of the data for training, 20% for validation.
"""
@author Rafael Diaz
@date 10/18/2020
"""
import toxic_word_collection as twc
import span_find as sf
from ast import literal_eval
import pandas as pd
import math
import os


def generate_spans_modified(read_file):
    """
    # modified version of generate spans from src/span_find.py.
    # this version shuffles the dataset before appending the toxic words to the data set
    to provide a different index for each line.
    :param read_file: This is the file to read data from, should be a csv
    :return: returns the length of 20% all text snippets for splitting data
    """
    trial = pd.read_csv(read_file)

    # trial = trial.sample(frac=1)
    trial['spans'] = trial.spans.apply(literal_eval)
    trial['toxic'] = [sf.extract_toxic_span(span_list, text) for span_list, text in zip(trial['spans'], trial['text'])]
    trial = trial.sample(frac=1)

    total_lines = 0
    for i in trial['spans']:
        total_lines += 1

    # print(total_lines)
    # print(math.ceil(total_lines * .2))

    path = os.path.join('data', 'randomized_data.csv')
    trial.to_csv(path)

    return math.ceil(total_lines * .2)


def split_file(read_file):
    """
    this program just splits the file. it takes the first 20% of the content and puts it into a file "evaluation_texts"
    and the other 80% are written into a file "training_texts"
    :param read_file:
    :return:
    """
    twent = generate_spans_modified(read_file)  # just assigning 20% of the length of the total text snippets to a
    # variable
    data = pd.read_csv(read_file)

    eval_path = os.path.join('data', 'evaluation_texts.csv')
    train_path = os.path.join('data', 'training_texts.csv')

    data[:twent].to_csv(eval_path)
    data[twent:].to_csv(train_path)


def create_toxic_dict(train_file):
    """
    creates a dictionary of all toxic words, each toxic word is the key. value assigned is 1
    :param train_file: this file should be the training_texts from the previous method.
    :return:
    """
    toxic_list = twc.quick_sort(twc.create_toxic_list(train_file))
    toxic_dict = {}
    for i in toxic_list:
        i = i.strip("\n")
        toxic_dict[i] = 1
    return toxic_dict


def compare_data(eval_path, train_path, write_path):
    """

    :param eval_path: evaluation texts from the previous method
    :param train_path: training texts from the previous method
    :param write_path: path in which the user would like to write the results to
    :return: master_list: 2d list of all indices
    """
    toxic_dict = create_toxic_dict(train_path)
    list_of_toxic = toxic_dict.keys()

    # list of spurious toxic words... words that have no "toxic" value
    ignore_list = ["hi", "his", "have", "for", "how", "know", "l", "like", "more", "nd", "saying", "something",
                   "t o", "than", "there", "the", "those", "to", "was", "we", "which", "with", "yet", "you",
                   "can", "could", "are", "that", "gonna", "only", "ose", "into", "imina", "but", "all", "look",
                   "and", "go", "many", "just", "is not", "what", "would", "get", "other", "others", "great",
                   "done", "big", "will", "under", "from", "funny", "about", "really", "another", "always",
                   "leg", "does", "been", "built", "art", "e he'", "because"]

    eval = pd.read_csv(eval_path)
    # print(list_of_toxic)
    indices = []
    master_list = []
    list_of_texts = [i for i in eval['text']]
    with open(write_path, 'w') as file:
        for text in list_of_texts:
            file.write(text + '\t')  # writes the text snippet into the file
            index_lists = "["
            for toxic in list_of_toxic:  # for loop which finds toxic words from dict in the text snippet
                if (toxic in text) and (toxic not in ignore_list):
                    index = text.index(toxic)  # gets the index of the first letter of the toxic word in the text
                    length_toxic = len(toxic)  # gets the length of the toxic word

                    # print('toxic word:', toxic)  # prints the actual toxic word from the list
                    # print('toxic from text:', text[index:index + length_toxic])  # prints the toxic word taken from
                    # the text snippet

                    for i in range(index, index + length_toxic):  # appends the indices to a list
                        if i not in indices:
                            # if a specific index is there, it doesn't append it, makes "idiot" turn to "idiots"
                            indices.append(i)

            indices.sort()
            master_list.append(indices)
            list_in_string = "["
            for i in indices:
                list_in_string += str(i) + ","
            file.write(list_in_string[:-1] + "]")  # writing the indices of the toxic words
            indices = []
            file.write('\n')
    return master_list


if __name__ == '__main__':
    path = os.path.join('data', 'combined_data_extracted.csv')
    train_path = os.path.join('data', 'training_texts.csv')
    eval_path = os.path.join('data', 'evaluation_texts.csv')
    comparison = os.path.join('data', 'compare.csv')  # path for where the file will be written
    generate_spans_modified(path)
    split_file(path)
    create_toxic_dict(train_path)
    print(compare_data(eval_path, train_path, comparison))

