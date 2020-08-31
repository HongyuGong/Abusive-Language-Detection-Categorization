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
import preprocessor as p
from param import *
import sys
import copy


def buildVocab(sentences):
  # Build vocabulary
  word_counts = Counter(itertools.chain(*sentences))
  # Mapping from index to word
  vocabulary_inv = [x[0] for x in word_counts.most_common()]
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


def buildVocabEmbed(train_data_path, test_data_path, vocab_pkl, embed_fn, embed_pkl):
    with open(train_data_path, "rb") as handle:
        train_sent_list, _ = pickle.load(handle)
    with open(test_data_path, "rb") as handle:
        test_sent_list, _ = pickle.load(handle)
    all_sent_list = list(train_sent_list) + list(test_sent_list) +  [["<PAD/>"]*1000]
    vocabulary, vocabulary_inv = buildVocab(all_sent_list)
    print("Vocabulary Size: {:d}".format(len(vocabulary)))
    with open(vocab_pkl, "wb") as handle:
        pickle.dump(vocabulary, handle)
    init_embed = getInitEmbed(vocabulary, embed_fn)
    with open(embed_pkl, "wb") as handle:
        pickle.dump(init_embed, handle)
    print("saving vocabulary and embedding...")
    

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
    parse.add_argument('--embed_fn', type=str, default=None, required=True,
                       help='the path of pre-trained embeddings')
    args = parser.parse_args()
    embed_fn = args.embed_fn
    
    train_data_path = os.path.join(param.dump_folder, "train_sent.data")
    test_data_path = os.path.join(param.dump_folder, "test_sent.data")
    vocab_pkl = os.path.join(param.dump_folder, "vocab.pkl")
    embed_pkl = os.path.join(param.dump_folder, "init_embed.pkl")
    norm_embed_pkl = os.path.join(param.dump_folder, "norm_init_embed.pkl")
    buildVocabEmbed(train_data_path, test_data_path, vocab_pkl, embed_fn, embed_pkl)
    normEmbed(embed_pkl, norm_embed_pkl)
