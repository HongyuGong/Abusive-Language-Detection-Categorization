"""
Preprocess the categorization dataset
"""

import numpy as np
import re
import itertools
import codecs
from collections import Counter
import csv
import pickle
import sys
import os
import copy
import editdistance

import data_util.param as param

def readCategoryData(fn):
    comment_list = []
    category_labels = []
    with open(fn, "rt") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        # skip reader
        next(reader, None)
        for row in reader:
            comment = row[0].strip()
            if (len(comment) == 0):
                continue
            bullying_label = int(row[2].strip().lower() == "yes")
            # explicit / implicit
            explicit_label = [int(row[3].strip().lower() == "yes"), int(row[1].strip().lower() == "yes")]
            # generic / targeted
            generic_label = [int(row[4].strip().lower() == "yes"), int(row[5].strip().lower() == "yes")]
            # gender
            val = int(row[8].strip().lower() == "yes")
            gender_label = [val, 1-val]
            # race
            val = int(row[9].strip().lower() == "yes")
            race_label = [val, 1-val]
            # appear
            val = int(row[7].strip().lower() == "yes")
            appear_label = [val, 1-val]
            # idea
            val = int(row[6].strip().lower() == "yes")
            idea_label = [val, 1-val]
            if (bullying_label < 1 or np.sum(explicit_label) < 1 or np.sum(generic_label) < 1):
                continue
            comment_list.append(" ".join(comment.split()))
            #comment_list.append(comment)
            category_labels.append((explicit_label[:], generic_label[:], gender_label[:], \
                                    race_label[:], appear_label[:], idea_label[:]))
    print("Comment # for categorization {}".format(len(comment_list)))
    return comment_list, category_labels


def readCommentData():
    with open(os.path.join(param.dump_folder, "raw_train_comm.data"), "rb") as handle:
        train_comments, train_labels = pickle.load(handle)
    with open(os.path.join(param.dump_folder, "raw_test_comm.data"), "rb") as handle:
        test_comments, test_labels = pickle.load(handle)
    with open(os.path.join(param.dump_folder, "train_map.data"), "rb") as handle:
        train_map_dict = pickle.load(handle)
    with open(os.path.join(param.dump_folder,"test_map.data"), "rb") as handle:
        test_map_dict = pickle.load(handle)

    abusive_train_comments = dict()
    for comm_ind, comm in enumerate(train_comments):
        if train_labels[comm_ind][0] == 1:
            comm = " ".join(comm.strip().split())
            abusive_train_comments[comm] = comm_ind

    abusive_test_comments = dict()
    for comm_ind, comm in enumerate(test_comments):
        if test_labels[comm_ind][0] == 1:
            comm = " ".join(comm.strip().split())
            abusive_test_comments[comm] = comm_ind
    return abusive_train_comments, abusive_test_comments, train_map_dict, test_map_dict


def dumpCategorizationData(fn="Anonymized_Comments_Categorized.csv"):
    # category data
    category_comments, category_labels = readCategoryData(fn)
    # original data
    train_comments_dict, test_comments_dict, train_map_dict, test_map_dict = readCommentData()
    print("Comment #: {}".format(len(train_comments_dict)+len(test_comments_dict)))

    category_match_dict = dict()
    orig_match_dict = dict()
    comm_category_dict = dict()
    match_count = 0
    for cate_comm_ind in range(len(category_comments)):
        # exact match
        cate_comm = category_comments[cate_comm_ind]
        is_match = False
        for train_comm in train_comments_dict:
            comm_ind = train_comments_dict[train_comm]
            if (cate_comm == train_comm):
                category_match_dict[cate_comm_ind] = ("train", comm_ind)
                if (("train", comm_ind) not in orig_match_dict):
                    orig_match_dict[("train", comm_ind)] = []
                orig_match_dict[("train", comm_ind)].append(cate_comm_ind)
                comm_category_dict[("train", comm_ind)] = category_labels[cate_comm_ind]
                is_match = True
                match_count += 1
                break
        if (is_match):
            continue
        for test_comm in test_comments_dict:
            comm_ind = test_comments_dict[test_comm]
            if (cate_comm == test_comm):
                category_match_dict[cate_comm_ind] = ("test", comm_ind)
                if (("test", comm_ind) not in orig_match_dict):
                    orig_match_dict[("test", comm_ind)] = []
                orig_match_dict[("test", comm_ind)].append(cate_comm_ind)
                comm_category_dict[("test", comm_ind)] = category_labels[cate_comm_ind]
                is_match = True
                match_count += 1
                break
        if (is_match):
            continue

    with open(os.path.join(param.dump_folder, "category_map.data"), "wb") as handle:
        pickle.dump(category_match_dict, handle)
    with open(os.path.join(param.dump_folder, "comm_category.data"), "wb") as handle:
        pickle.dump(comm_category_dict, handle)
    print("# of matched category comments:", len(category_match_dict))
    print("match count:", match_count)
    

if __name__=="__main__":
    dumpCategorizationData(param.categorization_dataset)



