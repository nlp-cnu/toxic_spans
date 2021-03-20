# this files goal is to convert our given data into a form compatible with BRAT (annotation device)
"""
@author rafaeldiaz
@date 03/20/2021
"""
import pandas as pd
import os
from ast import literal_eval


def convert_input(read_file, out_filename):
    # creating the path for reading data in and both output files, .txt and .ann
    read_path = os.path.join("data", read_file + ".csv")
    txt_file = os.path.join("data", out_filename + ".txt")
    ann_file = os.path.join("data", out_filename + ".ann")
    # creating a pandas frame of the read data
    data = pd.read_csv(read_path)
    # writing only the text column of the frame into the .txt file
    data['text'].to_csv(txt_file, index=False, header=False)

    # now a focus on creating the annotation file

    tcc = 0  # total character count
    ln = 1  # new line count (ROUGH ESTIMATE FOR TESTING)
    t = 1  # to be combined with the counter list
    counter = []  # format of T{count}
    types = []  # either gold or predicted
    indices = []  # start and end index of toxic word
    toxic_words = []  # just the toxic word
    separated_spans = []  # spans separated by consecutiv(ity)
    all_toxic = []  # all the toxic words but in a 2d list

    data['spans'] = data.spans.apply(literal_eval)
    for count, i in enumerate(data['spans']):  # count is the corresponding text number, i is the list of toxic spans

        if len(i) == 0:  # if the list is empty, append the text number with an empty list
            separated_spans.append([count, i])

        elif i[-1] - i[0] != len(i) - 1:  # checks if the list of numbers is consecutive throughout
            first = 0  # need to initialize variable early - used to mark the beginning of new set
            for index, item in enumerate(i):  # index is the index of span list, item is the span
                if index == 0:  # skip the first index since i search for the one behind
                    continue
                if i[index] - i[index - 1] > 1:  # if the difference between two pairs is more than 1, they are not
                    # consecutive
                    separated_spans.append([count, i[first] + tcc, i[index - 1] + tcc])  # append the first index and
                    # the previous
                    first = index  # new first index is the one that broke consecutive-ness
                if index == len(i) - 1:  # if this is the last index, append current first with last
                    separated_spans.append([count, i[first] + tcc, i[index] + tcc])
        else:
            separated_spans.append([count, i[0] + tcc, i[-1] + tcc])  # else it's all consecutive, append first and last

        tcc = tcc + len(data['text'][count]) + ln  # tcc is equal to it's current plus length of all chars in line
        ln = ln + 1  # add a new line cause next line (TESTING)

    for i in data['toxic']:
        if pd.isna(i):  # if whatever is in toxic at i is nan, append an empty string
            all_toxic.append([""])
        else:  # else we append the actual list ['moron', 'bigot']
            all_toxic.append(literal_eval(i))

    for array in all_toxic:  # for every list in the 2d list of toxic words (different text is a diff list)
        for i in range(len(array)):  # up to the amount of objects in that 2nd list
            currlist = separated_spans.pop(0)  # remove first list and save it to current
            if len(currlist) < 3:  # if the length is less than 3, it was an empty annotation -> that nan
                continue  # so we skip
            counter.append("T" + str(t))  # append the T1
            types.append("gold")  # it is gold because this is what we aim for
            indices.append([currlist[1], currlist[2]])  # appending the two indices from that list we popped
            toxic_words.append(array[i])  # append the current toxic word to a 1d list for writing purposes
            t = t + 1  # now we move onto the next toxic word

    with open(ann_file, 'w') as an:  # open the annotation file
        for i in range(len(counter)):  # basically get the index of all the lists at once
            an.write(counter[i] + "\t" + types[i] + " " + str(indices[i][0]) + " " + str(indices[i][1] + 1) + "\t" +
                     toxic_words[i] + "\n")
            # writing as...
            # Tx    gold index0 index1  toxic_word


if __name__ == "__main__":
    in_file = 'testing_file'
    out = 'testing'
    convert_input(in_file, out)