"""
Preprocess the classification dataset
"""

import numpy as np
import re
import itertools
import codecs
from collections import Counter
import csv
import pickle
import preprocessor as p
import sys
import copy

import data_util.param as param
import CMUTweetTagger
import CMUTweetTokenizer


"""
Preprocess texts
"""

def tok_str(string):
  string = re.sub(r"[^A-Za-z0-9()$,!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()


def readRawCommSentData(fn, verbose=True):
    with open(fn, "rt") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader, None)
        prev_comment_ind = 1  
        # [["this is true", "I have to say"], ["you hear that?", "alright", "forget it"]]
        comm_sent_list = []
        # [[0, 0], [0, 0, 0]]
        comm_label_list = []
        tmp_sent_list = []
        tmp_label_list = []
        for row in reader:
            comment_ind, _, sent, _, sent_bullying_label_str, _, _, _, _ = row
            try:
                comment_ind = int(comment_ind)
            except:
                print("empty comment_ind", row)
                continue
            sent = sent.strip()
            # skip empty sentence
            if (sent == ""):
                continue
            # sentence label
            if (sent_bullying_label_str.strip() == "No"):
                sent_label = 0
            elif (sent_bullying_label_str.strip() == "Yes"):
                sent_label = 1
            else:
                print("error in sent-level data: label str is neither yes nor no")
                print("row:", row)
                continue
            if (comment_ind == prev_comment_ind):
                tmp_sent_list.append(sent)
                tmp_label_list.append(sent_label)
            else:
                # store prev comment
                comm_sent_list.append(tmp_sent_list)
                comm_label_list.append(tmp_label_list)
                # update tmp
                tmp_sent_list = [sent]
                tmp_label_list = [sent_label]
                prev_comment_ind = comment_ind
        # store the last comment
        comm_sent_list.append(tmp_sent_list[:])
        comm_label_list.append(tmp_label_list[:])
        
        # sanity check: print last two comments and labels
        if verbose:
            for i in range(1, 3):
                print("example comment:", comm_sent_list[-i], "labels:", comm_label_list[-i])

        return comm_sent_list, comm_label_list


def shuffleCommData(comm_sent_list, comm_label_list, train_test_ratio):
    total_num = len(comm_sent_list)
    sent_arr = np.array(comm_sent_list)
    label_arr = np.array(comm_label_list)
    shuffle_indices = np.random.permutation(np.arange(total_num))
    shuffle_sent = sent_arr[shuffle_indices]
    shuffle_label = label_arr[shuffle_indices]
    # train data
    train_num = int(total_num * train_test_ratio)
    train_comm_list = shuffle_sent[:train_num]
    train_label_list = shuffle_label[:train_num]
    # test data
    test_comm_list = shuffle_sent[train_num:]
    test_label_list = shuffle_label[train_num:]
    return train_comm_list, train_label_list, test_comm_list, test_label_list


def mapCommSent(comm_sent_list):
    # comm-sent mapping
    map_dict = dict()
    start_ind = 0
    for comm_ind in range(len(comm_sent_list)):
        end_ind = start_ind + len(comm_sent_list[comm_ind]) # not including itself
        map_dict[comm_ind] = [start_ind, end_ind]
        start_ind = end_ind
    return map_dict


def vectorizeLabel(label_list):
    binary_label_list = [[label, 1-label] for label in label_list]
    return binary_label_list


def preprocessData(fn, save_fn):
    print(f"Preprocessing {fn}...")
    with open(os.path.join(param.dump_folder, fn), "rb") as handle:
        sent_list, label_list = pickle.load(handle)
    print(f"Tokenization...")
    sent_list =  [tok_str(sent) for sent in sent_list]
    # sent_list: a list of a list of tokens
    sent_list = genTokens(sent_list)
    with open(os.path.join(param.dump_folder, save_fn), "wb") as handle:
        pickle.dump((sent_list, label_list), handle)
    print(f"Done preprocessing, save to {save_fn}")
    

def getAnnotatedAttention(sent_fn, map_fn, attention_fn):
    attention_list = []
    with open(os.path.join(param.dump_folder, sent_fn), "rb") as handle:
        sent_list, sent_label_list = pickle.load(handle)
    with open(os.path.join(param.dump_folder, map_fn), "rb") as handle:
        map_dict = pickle.load(handle)
    for comm_ind in map_dict:
        sent_start_ind, sent_end_ind = map_dict[comm_ind]
        attention_vec = []
        for sent_ind in range(sent_start_ind, sent_end_ind):
            attention_vec = attention_vec + [sent_label_list[sent_ind][0]] * len(sent_list[sent_ind])
        attention_list.append(attention_vec[:])
    with open(os.path.join(param.dump_folder, attention_fn), "wb") as handle:
        pickle.dump(attention_list, handle)


"""
Preprocess POS tags
"""

def genTokens(sentences):
    # tokenize with tweet tokenizer
    tok_sentences = []
    max_sent_num = 5000
    ind = 0
    while (ind < len(sentences)):
        sent_list = sentences[ind:ind+max_sent_num]
        tok_sent_list = CMUTweetTokenizer.runtokenizer_parse(sent_list)
        tok_sentences = tok_sentences + tok_sent_list
        ind += max_sent_num
    # lower and split tokens
    tok_sentences = [sent.lower().split(" ") for sent in tok_sentences]
    print("# of tokenized sentences:", len(tok_sentences))
    return tok_sentences


def genPOSTags(text_path, pos_path, verbose=True):
    with open(text_path, "rb") as handle:
        comment_list, _ = pickle.load(handle)
    pos_comment_list = []
    max_sent_num = 5000
    ind = 0
    while (ind < len(comment_list)):
        sent_list = comment_list[ind: ind+max_sent_num]
        sent_list = [" ".join(seq) for seq in sent_list]
        raw_pos_list = CMUTweetTagger.runtagger_parse(sent_list)
        pos_list = []
        for raw_seq in raw_pos_list:
            seq = [tup[1] for tup in raw_seq]
            pos_list.append(seq[:])
        pos_comment_list = pos_comment_list + pos_list
        ind += max_sent_num
    with open(pos_path, "wb") as handle:
      pickle.dump(pos_comment_list, handle)
    if verbose:
      print("# pos sequences:", len(pos_comment_list))
      for i in range(3):
        print("example of pos sequence:", pos_comment_list[i])
      print(f"save pos to {pos_path}")
  
    
def dumpClassificationData(fn="Anonymized_Sentences_Classified.csv"):
    """
    data type:
    comm_list, sent_list, map_dict
    """
    comm_sent_list, comm_label_list = readRawCommSentData(fn)
    train_comm_sent_list, train_label_list, test_comm_sent_list, test_label_list \
                     = shuffleCommData(comm_sent_list, comm_label_list, train_test_ratio=0.8)
    train_map_dict = mapCommSent(train_comm_sent_list)
    test_map_dict = mapCommSent(test_comm_sent_list)
    
    # raw_train_comm_data
    merged_train_comm_list = [" ".join(sent_list) for sent_list in train_comm_sent_list]
    print("debug: comm_label_list", comm_label_list[:5])
    merged_train_label_list = [np.max(label_list) for label_list in train_label_list]
    merged_train_label_list = vectorizeLabel(merged_train_label_list)
    with open(os.path.join(param.dump_folder, "raw_train_comm.data"), "wb") as handle:
        pickle.dump((merged_train_comm_list, merged_train_label_list), handle)       
    # raw_train_sent_data
    flat_train_sent_list = list(itertools.chain.from_iterable(train_comm_sent_list))
    flat_train_label_list = list(itertools.chain.from_iterable(train_label_list))
    flat_train_label_list = vectorizeLabel(flat_train_label_list)
    with open(os.path.join(param.dump_folder, "raw_train_sent.data"), "wb") as handle:
        pickle.dump((flat_train_sent_list, flat_train_label_list), handle)
    with open(os.path.join(param.dump_folder, "train_map.data"), "wb") as handle:
        pickle.dump(train_map_dict, handle)

    # raw_test_comm_data
    merged_test_comm_list = [" ".join(sent_list) for sent_list in test_comm_sent_list]
    merged_test_label_list = [np.max(label_list) for label_list in test_label_list]
    merged_test_label_list = vectorizeLabel(merged_test_label_list)
    with open(os.path.join(param.dump_folder, "raw_test_comm.data"), "wb") as handle:
        pickle.dump((merged_test_comm_list, merged_test_label_list), handle)
    # raw_test_sent_data
    flat_test_sent_list = list(itertools.chain.from_iterable(test_comm_sent_list))
    flat_test_label_list = list(itertools.chain.from_iterable(test_label_list))
    flat_test_label_list = vectorizeLabel(flat_test_label_list)
    with open(os.path.join(param.dump_folder, "raw_test_sent.data"), "wb") as handle:
        pickle.dump((flat_test_sent_list, flat_test_label_list), handle)
    with open(os.path.join(param.dump_folder, "test_map.data"), "wb") as handle:
        pickle.dump(test_map_dict, handle)

    print("# of train comm: {}, # of test comm: {}".format(len(merged_train_comm_list), len(merged_test_comm_list)))
    print("# of train sent: {}, # of test sent: {}".format(len(flat_train_sent_list), len(flat_test_sent_list)))

    # preprocessing word sequence
    fn_list = ["raw_train_comm.data", "raw_train_sent.data", \
               "raw_test_comm.data", "raw_test_sent.data"]
    save_fn_list = ["train_comm.data", "train_sent.data", \
                    "test_comm.data", "test_sent.data"]
    for (fn, save_fn) in zip(fn_list, save_fn_list):
        preprocessData(fn, save_fn)

    # get attention
    getAnnotatedAttention("train_sent.data", "train_map.data", "train_attention.data")
    getAnnotatedAttention("test_sent.data", "test_map.data", "test_attention.data")

    # get POS tags
    fn_list = ["train_comm.data", "test_comm.data"]
    save_fn_list = ["train_comm_pos.data", "test_comm_pos.data"]
    for (fn, save_fn) in zip(fn_list, save_fn_list):
      genPOSTags(fn, save_fn)
  

if __name__=="__main__":

    # dump preprocessed comments and sentences
    dumpClassificationData(param.classification_dataset)


    









