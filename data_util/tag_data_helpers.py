"""
POS tag helpers for model training
"""
import numpy as np
import data_util.param as param

def padPOSSents(pos_sentences, max_len, padding_pos=param.pad):
    length_list = []
    padded_pos_sentences = []
    for i in range(len(pos_sentences)):
        sent = pos_sentences[i][:max_len]
        num_padding = max_len - len(sent)
        new_sentence = sent + [padding_pos] * num_padding
        length_list.append(len(new_sentence))
        padded_pos_sentences.append(new_sentence[:])
    return padded_pos_sentences, np.array(length_list)


def cleanPOSSents(pos_sentences, pos_vocabulary, unk_pos=param.unk):
    # replace pos tags not in pos_vocabulary with unk
    for (sent_ind, pos_sent) in enumerate(pos_sentences):
        for (word_ind, word) in enumerate(pos_sent):
            if word not in pos_vocabulary:
                pos_sentences[sent_ind][word_ind] = unk_pos


def genPOSFeatures(pos_sentences, max_sent_len, pos_vocabulary, verbose=True):
    padded_pos_sentences, length_list = padPOSSents(pos_sentences, max_sent_len)
    cleanPOSSents(padded_pos_sentences, pos_vocabulary)
    x = np.array([[pos_vocabulary[word] for word in sent] for sent in padded_pos_sentences])
    if verbose:
        print("padded pos sentences:", np.array(padded_pos_sentences).shape)
        print("debug padded_pos_sentences:", padded_pos_sentences[0][:10])
        print("pos feature shape:", np.array(x).shape)
    return x, length_list
