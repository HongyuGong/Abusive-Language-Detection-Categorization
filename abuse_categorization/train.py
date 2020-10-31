"""
Train abusive language categorization model
"""

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os
import sys
import time
import datetime
import pickle
import copy
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import data_util.param as param
from data_util import data_helpers, tag_data_helpers, category_data_helpers
from model.abuse_categorizer import AbuseCategorizer

tf.set_random_seed(111)
# multitask setting
tf.flags.DEFINE_boolean("multitask", True, "use multitasking for model training")
tf.flags.DEFINE_integer("primary_label", 0, "primary label in multitasking (0: gender, 1: race, 2: race, 3: ideology")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("pos_vocab_size", 25, "Vocab size of POS tags")
tf.flags.DEFINE_integer("pos_embedding_dim", 25, "Dcimensionality of pos tag embedding (default: 20)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
#tf.flags.DEFINE_boolean("use_attention", True, "whether to add supervised attention")
tf.flags.DEFINE_float("attention_lambda", 0.2, "Supervised attention lambda (default: 0.05)")
tf.flags.DEFINE_string("attention_loss_type", 'encoded', "loss function of attention")
tf.flags.DEFINE_float("l2_reg_lambda", 0.02, "L2 regularizaion lambda (default: 0.05)")
tf.flags.DEFINE_integer("hidden_size", 300, "Dimensionality of RNN cell (default: 300)")
tf.flags.DEFINE_integer("pos_hidden_size", 25, "Dimensionality of POS-RNN cell")
tf.flags.DEFINE_integer("attention_size", 20, "Dimensionality of attention scheme (default: 50)")
tf.flags.DEFINE_boolean("use_pos_flag", True, "use the sequence of POS tags")
# Training parameters -- evaluate_every:100
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_string("checkpoint", '', "model")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
"""
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
"""

# single-/multi-tasking parameters
if (FLAGS.multitask):
    category_weights = [0.1, 0.1, 0.1, 0.1]
    category_weights[FLAGS.primary_label] = 0.7
else:
    category_weights = [0.0, 0.0, 0.0, 0.0]
    category_weights[FLAGS.primary_label] = 1.0

# -----------------------------  load data  -----------------------------
vocabulary, pos_vocabulary, init_embed = data_helpers.loadVocabEmb()
pos_vocab_size = len(pos_vocabulary)
x_train, length_train, attention_train, pos_train, pos_length_train, \
         train_gender_labels, train_race_labels, train_appear_labels, train_idea_labels, \
         x_dev, length_dev, attention_dev, pos_dev, pos_length_dev, \
         dev_gender_labels, dev_race_labels, dev_appear_labels, dev_idea_labels, \
         = category_data_helpers.loadCategoryTrainData()

# ----------------------------- model training --------------------------
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        model = AbuseCategorizer(
            max_sequence_length = param.max_sent_len,
            num_classes = 2,
            pos_vocab_size = pos_vocab_size,
            init_embed = init_embed,
            hidden_size = FLAGS.hidden_size,
            attention_size = FLAGS.attention_size,
            keep_prob = FLAGS.dropout_keep_prob,
            attention_lambda = FLAGS.attention_lambda,
            attention_loss_type = FLAGS.attention_loss_type,
            l2_reg_lambda = FLAGS.l2_reg_lambda,
            use_pos_flag = FLAGS.use_pos_flag,
            category_weights = category_weights)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss, aggregation_method=2)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # save models
        if FLAGS.checkpoint == "":
            out_dir = os.path.abspath(os.path.join(os.path.pardir, "model"))
            print("Writing to {}\n".format(out_dir))
        else:
            out_dir = FLAGS.checkpoint
        if (FLAGS.multitask):
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "multitask_attn="+FLAGS.attention_loss_type+"_lambda="+str(FLAGS.attention_lambda)))
        else:
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "singletask_attn="+FLAGS.attention_loss_type+"_lambda="+str(FLAGS.attention_lambda)))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())
        # initalize variables
        sess.run(tf.initialize_all_variables())
        """
        # resotre models
        try:
            saver.restore(sess, os.path.join(checkpoint_dir, "best_model"))
            print("restoring from trained model...")
        except:
            print("train a new model...")
        """

        def train_step(x_batch, pos_batch, attention_batch, y_gender_batch, y_race_batch, y_appear_batch, y_idea_batch, \
                       sequence_length, pos_sequence_length):
            feed_dict = {
                model.input_x: x_batch,
                model.input_pos: pos_batch, 
                model.sequence_length: sequence_length,
                model.input_attention: attention_batch,
                model.input_y_gender: y_gender_batch,
                model.input_y_race: y_race_batch,
                model.input_y_appear: y_appear_batch,
                model.input_y_idea: y_idea_batch,
                model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
            _, step, loss = sess.run(
                [train_op, global_step, model.loss],
                feed_dict)
            if (step % FLAGS.evaluate_every == 0):
                print("step {}, loss {:} ".format(step, loss))


        def eval_single_label(gold_labels, dev_scores):
            # gold single scores
            gold_scores = [labels[0] for labels in gold_labels]
            # pred single scores
            pred_scores = [scores[0] for scores in dev_scores]
            # pr auc
            prec, recall, _ = precision_recall_curve(gold_scores, pred_scores, pos_label=1)
            pr_auc = auc(recall, prec)
            return pr_auc
            

        def dev_step(x_dev, pos_dev, length_dev, pos_length_dev, attention_dev, y_gender_dev, y_race_dev, \
                     y_appear_dev, y_idea_dev, writer=None):
            dev_gender_scores = []
            dev_race_scores = []
            dev_appear_scores = []
            dev_idea_scores = []
            pos = 0
            gap = 50
            loss_list = []
            while (pos < len(x_dev)):
                x_batch = x_dev[pos:pos+gap]
                pos_batch = pos_dev[pos:pos+gap]
                sequence_length = length_dev[pos:pos+gap]
                pos_sequence_length = pos_length_dev[pos:pos+gap]
                attention_batch = attention_dev[pos:pos+gap]
                y_gender_batch = y_gender_dev[pos:pos+gap]
                y_race_batch = y_race_dev[pos:pos+gap]
                y_appear_batch = y_appear_dev[pos:pos+gap]
                y_idea_batch = y_idea_dev[pos:pos+gap]
                pos += gap
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_pos: pos_batch,
                    model.input_attention: attention_batch,
                    model.input_y_gender: y_gender_batch,
                    model.input_y_race: y_race_batch,
                    model.input_y_appear: y_appear_batch,
                    model.input_y_idea: y_idea_batch,
                    model.sequence_length: sequence_length,
                    model.dropout_keep_prob: 1.0
                    }
                step, loss, scores_gender, scores_race, scores_appear, scores_idea = sess.run(
                    [global_step, model.loss, model.prob_gender, model.prob_race, model.prob_appear, model.prob_idea], feed_dict)
                dev_gender_scores = dev_gender_scores + list(scores_gender)
                dev_race_scores = dev_race_scores + list(scores_race)
                dev_appear_scores = dev_appear_scores + list(scores_appear)
                dev_idea_scores = dev_idea_scores + list(scores_idea)
                loss_list.append(loss)
            # evaluation
            gender_pr_auc = eval_single_label(y_gender_dev, dev_gender_scores)
            race_pr_auc = eval_single_label(y_race_dev, dev_race_scores)
            appear_pr_auc = eval_single_label(y_appear_dev, dev_appear_scores)
            idea_pr_auc = eval_single_label(y_idea_dev, dev_idea_scores)
            print("PR AUC:\n,gender: {}, race: {}, appearance: {}, ideology: {}".format(gender_pr_auc, race_pr_auc, appear_pr_auc, idea_pr_auc))
            avg_loss = np.mean(loss_list)
            print("dev loss:", avg_loss)
            return gender_pr_auc, race_pr_auc, appear_pr_auc, idea_pr_auc, avg_loss


        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, length_train, pos_train, pos_length_train, attention_train, train_gender_labels, train_race_labels, \
                     train_appear_labels, train_idea_labels)), FLAGS.batch_size, FLAGS.num_epochs)
        best_auc = 0.0
        best_loss = 100.0
        for batch in batches:
            x_batch, length_batch, pos_batch, pos_length_batch, attention_batch, y_gender_batch, y_race_batch, \
                     y_appear_batch, y_idea_batch = zip(*batch)
            train_step(x_batch, pos_batch, np.array(attention_batch), y_gender_batch, y_race_batch, y_appear_batch, y_idea_batch, \
                       length_batch, pos_length_batch)
            current_step = tf.train.global_step(sess, global_step)
            if (current_step% FLAGS.evaluate_every == 0):
                print("\n Evaluation:")
                gender_pr_auc, race_pr_auc, appear_pr_auc, idea_pr_auc, avg_loss = \
                               dev_step(x_dev, pos_dev, length_dev, pos_length_dev, attention_dev, dev_gender_labels, dev_race_labels, \
                                        dev_appear_labels, dev_idea_labels)
                pr_auc_list = [gender_pr_auc, race_pr_auc, appear_pr_auc, idea_pr_auc]
                pr_auc = pr_auc_list[FLAGS.primary_label]
                #if ((not is_multi) and best_auc < pr_auc):
                if ( best_auc < pr_auc):
                    best_auc = pr_auc
                    print("best pr_auc:", best_auc)
                    checkpoint_prefix = os.path.join(checkpoint_dir, "best_model")
                    path = saver.save(sess, checkpoint_prefix)
                    print("Saved best model checkpoint.")
