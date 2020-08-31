"""
save tag sequence for bullying detection
"""

import os
import pickle
import numpy as np
import data_helpers
import sys
sys.path.append("../")
import CMUTweetTagger
import CMUTweetTokenizer


dump_folder = "dump/"

def sanityCheck(text_path):
    # check whether sentences are empty
    with open(text_path, "rb") as handle:
        train_texts, _ = pickle.load(handle)
    empty_num = 0
    for seq in train_texts:
        try:
            if (len(seq) == 0):
                empty_num += 1
        except:
            print("seq", seq)
    print("empty num:", empty_num)


def filterData(text_path, filtered_text_path):
    # remove empty comments from dataset
    with open(text_path, "rb") as handle:
        comment_list, label_list = pickle.load(handle)
    invalid_inds = [ind for ind in range(len(comment_list)) if len(comment_list[ind]) == 0]
    comment_list = [comment_list[ind] for ind in range(len(comment_list)) if ind not in invalid_inds]
    label_list = [label_list[ind] for ind in range(len(label_list)) if ind not in invalid_inds]
    print("len comment_list:", len(comment_list), "len label_list:", len(label_list))
    with open(filtered_text_path, "wb") as handle:
        pickle.dump((comment_list, label_list), handle)


def genTokens(sentences):
    tok_sentences = []
    max_sent_num = 5000
    ind = 0
    while (ind < len(sentences)):
        sent_list = sentences[ind:ind+max_sent_num]
        tok_sent_list = CMUTweetTokenizer.runtokenizer_parse(sent_list)
        tok_sentences = tok_sentences + tok_sent_list
        ind += max_sent_num
        #print("sent_list:", sent_list[0])
        #print("tok_sent_list:", tok_sent_list[0])
    # lower and split tokens
    tok_sentences = [sent.lower().split(" ") for sent in tok_sentences]
    print("# of tokenized sentences:", len(tok_sentences))
    return tok_sentences


def genPOSTags(text_path, pos_save_path):
    with open(text_path, "rb") as handle:
        raw_comment_list, _ = pickle.load(handle)
    print("# of comments:", len(raw_comment_list))
    # CMUTokenizer
    comment_list = genTokens(raw_comment_list)
    pos_comment_list = []
    max_sent_num = 5000
    ind = 0
    while (ind < len(comment_list)):
        # sent format: ["example tweet"]
        sent_list = comment_list[ind: ind+max_sent_num]
        # convert seq to str
        sent_list = [" ".join(seq) for seq in sent_list]
        # pos_sent format: [[("example", "N", 0.97), ("tweet", "N", 0.7)], [], ...]
        raw_pos_list = CMUTweetTagger.runtagger_parse(sent_list)
        # refine pos sequence: [["N", "N"], ...]
        pos_list = []
        for raw_seq in raw_pos_list:
            seq = [tup[1] for tup in raw_seq]
            pos_list.append(seq[:])
        pos_comment_list = pos_comment_list + pos_list
        ind += max_sent_num
    print("# pos sequences:", len(pos_comment_list))
    print("example of pos sequence:", pos_comment_list[0])
    print("example of pos sequence:", pos_comment_list[-1])
    # save pos sequence
    if (pos_save_path != None):
        with open(pos_save_path, "wb") as handle:
            pickle.dump(pos_comment_list, handle)
        print("saving pos tags...")
    return comment_list, pos_comment_list
    

"""
utility functions to build vocab and pad pos-sentences
"""

def getPOSVocab(train_pos_sentences, test_pos_sentences):
    all_pos_sentences = list(train_pos_sentences) + list(test_pos_sentences) + [["<POS/>"]*1000]
    vocabulary, vocabulary_inv = comm_sent_data_helpers.buildVocab(all_pos_sentences)
    print("POS vocabulary size:", len(vocabulary))
    with open(dump_folder+"pos_vocab.pkl", "wb") as handle:
        pickle.dump(vocabulary, handle)
    for w in vocabulary:
        print("example vocab:", w, vocabulary[w])
        #break
    print("saving pos vocabulary...")
    


def padPOSSents(pos_sentences, max_len, padding_pos="<POS/>"):
    #length_list = np.array([len(sent) for sent in pos_sentences])
    length_list = []
    padded_pos_sentences = []
    for i in range(len(pos_sentences)):
        sent = pos_sentences[i][:max_len]
        num_padding = max_len - len(sent)
        new_sentence = sent + [padding_pos] * num_padding
        length_list.append(len(new_sentence))
        padded_pos_sentences.append(new_sentence[:])
    return padded_pos_sentences, np.array(length_list)
        

def genPOSFeatures(pos_sentences, max_sent_len, pos_vocabulary):
    #pad comments
    padded_pos_sentences, length_list = padPOSSents(pos_sentences, max_sent_len)
    print("padded pos sentences:", np.array(padded_pos_sentences).shape)
    print("debug padded_pos_sentences:", padded_pos_sentences[0][:10])
    x = np.array([[pos_vocabulary[word] for word in sent] for sent in padded_pos_sentences])
    #print("debug pos feature", x[0][:10])
    print("pos feature shape:", np.array(x).shape)
    return x, length_list
    

if __name__=="__main__":
    # comment pos
    train_text_path = dump_folder+"raw_train_comm.data"
    train_pos_path = dump_folder+"train_comm_pos.data"
    genPOSTags(train_text_path, train_pos_path)

    test_text_path=dump_folder+"raw_test_comm.data"
    test_pos_path = dump_folder+"test_comm_pos.data"
    genPOSTags(test_text_path, test_pos_path)
    
    # sentence pos
    train_text_path = dump_folder+"raw_train_sent.data"
    train_pos_path = dump_folder+"train_sent_pos.data"
    genPOSTags(train_text_path, train_pos_path)

    test_text_path=dump_folder+"raw_test_sent.data"
    test_pos_path = dump_folder+"test_sent_pos.data"
    genPOSTags(test_text_path, test_pos_path)

    # get pos vocabulary
    train_pos_path = dump_folder+"train_comm_pos.data"
    with open(train_pos_path, "rb") as handle:
        train_pos_sentences = pickle.load(handle)
    test_pos_path = dump_folder+"test_comm_pos.data"
    with open(test_pos_path, "rb") as handle:
        test_pos_sentences = pickle.load(handle)
    getPOSVocab(train_pos_sentences, test_pos_sentences)
    


    



    
