"""
Data helpers for model training
"""
import numpy as np
import re
import os
import itertools
import codecs
from collections import Counter
import csv
import pickle
import preprocessor as p
import sys
import copy

import data_util.param as param
from data_util import data_helpers, tag_data_helpers


def countLabelStats(bin_labels, bin_names):
    num = len(bin_labels)
    pos_num = np.sum([labels[0] for labels in bin_labels])
    pos_ratio = float(pos_num) / num
    print("total num: {}, {} ratio: {}, {} ratio: {}".format(num, bin_names[0], pos_ratio, bin_names[1], 1-pos_ratio))

    
def loadCategoryData(data_type, verbose=True):
    x, length, attention, pos, pos_length, _ = data_helpers.loadData(data_type, verbose)
    with open(os.path.join(param.dump_folder, "comm_category.data"), "rb") as handle:
        comm_category_map = pickle.load(handle)
    cate_x, cate_length, cate_attention, cate_pos, cate_pos_length  = [], [], [], [], []
    gender_labels, race_labels, appear_labels, idea_labels = [], [], [], []
    # category idx
    explicit_idx, generic_idx, gender_idx, race_idx, appear_idx, idea_idx  = list(range(6))

    for key in comm_category_map:
        dt, comm_ind = key
        cate_labels = comm_category_map[key]
        if dt == data_type:
            cate_x.append(copy.deepcopy(x[comm_ind]))
            cate_length.append(copy.deepcopy(length[comm_ind]))
            cate_attention.append(copy.deepcopy(attention[comm_ind]))
            cate_pos.append(copy.deepcopy(pos[comm_ind]))
            cate_pos_length.append(copy.deepcopy(pos_length[comm_ind]))
            # category label
            gender_labels.append(copy.deepcopy(cate_labels[gender_idx]))
            race_labels.append(copy.deepcopy(cate_labels[race_idx]))
            appear_labels.append(copy.deepcopy(cate_labels[appear_idx]))
            idea_labels.append(copy.deepcopy(cate_labels[idea_idx]))
        else:
            continue 
    if verbose:
        print("{} example #: {} for categorization".format(data_type, len(cate_x)))
        countLabelStats(gender_labels, ["gender", "non-gender"])
        countLabelStats(race_labels, ["race", "non-race"])
        countLabelStats(appear_labels, ["appear", "non-appear"])
        countLabelStats(idea_labels, ["idea", "non-idea"])
        
    return cate_x, cate_length, cate_attention, cate_pos, cate_pos_length, \
           gender_labels, race_labels, appear_labels, idea_labels


def loadCategoryTrainData():
    data = loadCategoryData(data_type="train", verbose=True)
    return data_helpers.splitTrainData(data, train_ratio=0.8, verbose=True)


def loadCategoryTestData():
    return loadCategoryData(data_type="test", verbose=True)



