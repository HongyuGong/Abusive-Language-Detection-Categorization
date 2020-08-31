"""
Build the vocabulary and embeddings
"""
import numpy as np
import re
import os
import argparse
import itertools
import codecs
from collections import Counter
import csv
import pickle
import sys
import copy

import data_util.param as param


def buildVocab(sentences, unk=param.unk, pad=param.pad):
  # Build vocabulary
  word_counts = Counter(itertools.chain(*sentences))
  # Mapping from index to word
  vocabulary_inv = [unk, pad] + [x[0] for x in word_counts.most_common()]
  # Mapping from word to index
  vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
  return [vocabulary, vocabulary_inv]


def getInitEmbed(vocabulary, embed_fn, word_limit=50000):
  f = open(embed_fn, "r")
  # parameters of word vectors
  header = f.readline()
  vocab_size, dim = np.array(header.strip().split(), "int")
  print("vector dim:", dim)

  all_vocab = []
  all_embed_arr = []
  i = 0
  while (i < min(word_limit, vocab_size)):
    line = f.readline()
    i += 1
    seq = line.strip().split()
    try:
        all_vocab.append(seq[0])
        vec = np.array(seq[1:], "float")
        all_embed_arr.append(vec[:])
    except:
        print("wrong line:", i)
        sys.exit(0)
  f.close()
  init_embed = []
  unknown_word = []
  for w in vocabulary:
    try:
      ind = all_vocab.index(w)
      vec = all_embed_arr[ind]
    except:
      vec = (np.random.rand(dim) - 0.5) * 2 # random vec generation [-1, 1]
      unknown_word.append(w)
    init_embed.append(vec[:])
  print("unknown word:", len(unknown_word), unknown_word[:10])
  init_embed = np.array(init_embed)
  print("init_embed shape", init_embed.shape)
  return init_embed


def savePOSVocab(data_path, vocab_pkl):
    with open(data_path, "rb") as handle:
        sent_list = pickle.load(handle)
    vocabulary, vocabulary_inv = buildVocab(sent_list)
    with open(vocab_pkl, "wb") as handle:
        pickle.dump(vocabulary, handle)
    print("Vocab size: {}, save to {}".format(len(vocabulary), vocab_pkl))


def saveVocabEmbed(data_path, vocab_pkl, embed_fn, embed_pkl):
    with open(data_path, "rb") as handle:
        sent_list, _ = pickle.load(handle)
    vocabulary, vocabulary_inv = buildVocab(sent_list)
    with open(vocab_pkl, "wb") as handle:
        pickle.dump(vocabulary, handle)
    print("Vocab size: {}, save to {}".format(len(vocabulary), vocab_pkl))
    init_embed = getInitEmbed(vocabulary, embed_fn)
    with open(embed_pkl, "wb") as handle:
        pickle.dump(init_embed, handle)
    print("Save embedding to {}".format(embed_pkl))

# use only train data!!!
# change genFeatures: replace test word with unk    

def normEmbed(embed_pkl, norm_embed_pkl):
    with open(embed_pkl, "rb") as handle:
        init_embed = pickle.load(handle)
    norm_init_embed = []
    for vec in init_embed:
        norm_vec = np.array(vec) / np.linalg.norm(vec)
        norm_init_embed.append(list(norm_vec)[:])
    norm_init_embed = np.array(norm_init_embed)
    with open(norm_embed_pkl, "wb") as handle:
        pickle.dump(norm_init_embed, handle)
    print("done saving normalized vectors.")

    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Prepare vocabulary and embedding.')
    parser.add_argument('--embed_fn', type=str, default=None, required=True,
                       help='the path of pre-trained embeddings')
    args = parser.parse_args()
    embed_fn = args.embed_fn
    
    data_path = os.path.join(param.dump_folder, "train_comm.data")
    vocab_pkl = os.path.join(param.dump_folder, "vocab.pkl")
    embed_pkl = os.path.join(param.dump_folder, "init_embed.pkl")
    norm_embed_pkl = os.path.join(param.dump_folder, "norm_init_embed.pkl")
    pos_data_path = os.path.join(param.dump_folder, "train_comm_pos.data")
    pos_vocab_pkl = os.path.join(param.dump_folder, "pos_vocab.pkl")

    # word vocab and embed
    saveVocabEmbed(data_path, vocab_pkl, embed_fn, embed_pkl)
    normEmbed(embed_pkl, norm_embed_pkl)
    
    # POS vocab
    savePOSVocab(pos_data_path, pos_vocab_pkl)
