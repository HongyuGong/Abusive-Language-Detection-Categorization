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
import sys
import copy

import data_util.param as param
from data_util import tag_data_helpers


def loadVocabEmb():
    # vocabulary & embedding
    with open(os.path.join(param.dump_folder, "vocab.pkl"), "rb") as handle:
        vocabulary = pickle.load(handle)
    with open(os.path.join(param.dump_folder, "pos_vocab.pkl"), "rb") as handle:
        pos_vocabulary = pickle.load(handle)
    with open(os.path.join(param.dump_folder, "norm_init_embed.pkl"), "rb") as handle:
        init_embed = pickle.load(handle)
    return vocabulary, pos_vocabulary, init_embed


def splitTrainData(data, train_ratio=0.8, verbose=True):
    # split train data into train & dev sets
    data_size = len(data[0])
    groups = len(data)
    train_size = int(data_size * train_ratio)
    train_inds = set(np.random.choice(range(data_size), size=train_size, replace=False))
    train_data = [[] for t in range(groups)]
    dev_data = [[] for t in range(groups)]
    for ind in range(data_size):
        if ind in train_inds:
            for t in range(groups):
                train_data[t].append(copy.deepcopy(data[t][ind]))
        else:
            for t in range(groups):
                dev_data[t].append(copy.deepcopy(data[t][ind]))
    if verbose:
        print("split into train ({} examples) and dev sets ({} examples)".format(len(train_data[0]), len(dev_data[0])))
    return train_data + dev_data
    

def loadData(data_type, verbose=True):
    assert data_type in ["train", "test"]
    with open(os.path.join(param.dump_folder, "vocab.pkl"), "rb") as handle:
        vocabulary = pickle.load(handle)
    with open(os.path.join(param.dump_folder, "pos_vocab.pkl"), "rb") as handle:
        pos_vocabulary = pickle.load(handle)
    with open(os.path.join(param.dump_folder, data_type+"_comm.data"), "rb") as handle:
        sentences, labels = pickle.load(handle)
    with open(os.path.join(param.dump_folder, data_type+"_comm_pos.data"), "rb") as handle:
        pos_sentences = pickle.load(handle)
    with open(os.path.join(param.dump_folder, data_type+"_attention.data"), "rb") as handle:
        attention = pickle.load(handle)
    # generate features & labels
    x, length, attention = genFeatures(sentences, attention, param.max_sent_len, vocabulary)
    pos, pos_length = tag_data_helpers.genPOSFeatures(pos_sentences, param.max_sent_len, pos_vocabulary)
    y = np.array(labels)
    if verbose:
        print("load {} data, input sent size: {}, input POS size: {}, label size: {}".format(
            data_type, np.array(x).shape, np.array(pos).shape, np.array(y).shape))
    return x, length, attention, pos, pos_length, y


def loadTrainData():
     data = loadData(data_type="train", verbose=True)
     return splitTrainData(data, train_ratio=0.8, verbose=True)


def loadTestData():
    return loadData(data_type="test", verbose=True)
    

def padSents(sentences, max_len, padding_word=param.pad):
    #length_list = np.array([len(sent) for sent in sentences])
    length_list = []
    padded_sentences = []
    for i in range(len(sentences)):
        sent = sentences[i][:max_len]
        num_padding = max_len - len(sent)
        new_sentence = sent + [padding_word] * num_padding
        length_list.append(len(new_sentence))
        padded_sentences.append(new_sentence)
    return padded_sentences, np.array(length_list)


def genFeatures(sent_list, attention_list, max_sent_len, vocabulary):
    # pad sentences
    padded_sent_list, length_list = padSents(sent_list, max_sent_len)
    padded_attention_list, _ = padSents(attention_list, max_sent_len, 0)
    print("padded sent:", np.array(padded_sent_list).shape)
    # generate features
    x = []
    for sent in padded_sent_list:
        sent_x = []
        for word in sent:
            try:
                sent_x.append(vocabulary[word])
            except:
                sent_x.append(vocabulary[param.unk])
                continue
        x.append(sent_x[:])
    x = np.array(x)
    #x = np.array([[vocabulary[word] for word in sent] for sent in padded_sent_list])
    padded_attention_list = np.array(padded_attention_list)
    print("feature shape:", np.array(x).shape)
    return x, length_list, padded_attention_list


def batch_iter(data, batch_size, num_epochs):
  """
  Generates a batch iterator for a dataset.
  """
  data = np.array(data)
  data_size = len(data)
  num_batches_per_epoch = int(data_size/batch_size) + 1
  for epoch in range(num_epochs):
    # Shuffle the data at each epoch
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_data = data[shuffle_indices]
    for batch_num in range(num_batches_per_epoch):
      start_index = batch_num * batch_size
      end_index = min((batch_num + 1) * batch_size, data_size)
      yield shuffled_data[start_index:end_index]


