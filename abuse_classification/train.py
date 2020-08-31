"""
Train abusive language classifier
"""

import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np
import os
import sys
import copy
import time
import datetime
import pickle

import param
from data_util import data_helpers, tag_data_helpers
from model.abuse_classifier import AbuseClassifier

tf.set_random_seed(1111)
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("pos_vocab_size", 25, "Vocab size of POS tags")
tf.flags.DEFINE_integer("pos_embedding_dim", 25, "Dimensionality of pos tag embedding (default: 20)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("attention_lambda", 0.2, "Supervised attention lambda (default: 0.05)")
tf.flags.DEFINE_string("attention_loss_type", 'hinge', "loss function of attention")
tf.flags.DEFINE_float("l2_reg_lambda", 0.02, "L2 regularizaion lambda (default: 0.05)")
tf.flags.DEFINE_integer("hidden_size", 300, "Dimensionality of RNN cell (default: 300)")
tf.flags.DEFINE_integer("pos_hidden_size", 25, "Dimensionality of POS-RNN cell")
tf.flags.DEFINE_integer("attention_size", 20, "Dimensionality of attention scheme (default: 50)")
tf.flags.DEFINE_boolean("use_pos_flag", True, "use the sequence of POS tags")
# Training parameters -- evaluate_every should be 100
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 60, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500000, "Save model after this many steps (default: 100)")
#tf.flags.DEFINE_float("train_ratio", 1.0, "Ratio of training data")
# Misc Parameters
tf.flags.DEFINE_string("checkpoint", '', "model")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# -----------------------------  load data  -----------------------------
vocabulary, pos_vocabulary, init_embed = data_helpers.loadVocabEmb()

x_train, length_train, attention_train, pos_train, pos_length_train, y_train, \
         x_dev, length_dev, attention_dev, pos_dev, pos_length_dev, y_dev \
         = data_helpers.loadTrainData()

# -------------------------- model training --------------------------
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        model = AbuseClassifier(
            max_sequence_length = param.max_sent_len,
            num_classes = 2,
            embedding_size = FLAGS.embedding_dim,
            pos_vocab_size = FLAGS.pos_vocab_size,
            pos_embedding_size = FLAGS.pos_embedding_dim,
            init_embed = init_embed,
            hidden_size = FLAGS.hidden_size,
            pos_hidden_size = FLAGS.pos_hidden_size,
            attention_size = FLAGS.attention_size,
            keep_prob = FLAGS.dropout_keep_prob,
            attention_lambda = FLAGS.attention_lambda,
            attention_loss_type = FLAGS.attention_loss_type,
            l2_reg_lambda = FLAGS.l2_reg_lambda,
            use_pos_flag = FLAGS.use_pos_flag)
        
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss, aggregation_method=2)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # save models
        if FLAGS.checkpoint == "":
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "model"))
            print("Writing to {}\n".format(out_dir))
        else:
            out_dir = FLAGS.checkpoint
        if (FLAGS.attention_lambda == 0.0):
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "model_noatt_checkpoints"))
        else:
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "model_att="+FLAGS.attention_loss_type+"_checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())
        # initalize variables
        sess.run(tf.initialize_all_variables())
        # resotre models
        #try:
        #    saver.restore(sess, os.path.join(checkpoint_dir, "best_model"))
        #    print("restoring from trained model...")
        #except:
        print("train a new model...")

        def train_step(x_batch, pos_batch, y_batch, sequence_length, pos_sequence_length, attention_batch):
            feed_dict = {
                model.input_x: x_batch,
                model.input_pos: pos_batch,
                model.input_y: y_batch,
                model.sequence_length: sequence_length,
                model.input_attention: attention_batch,
                model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
            _, step, loss = sess.run(
                [train_op, global_step, model.loss],
                feed_dict)
            if (step % FLAGS.evaluate_every == 0):
                print("step {}, loss {:} ".format(step, loss))

        def dev_step(x_dev, pos_dev, y_dev, length_dev, pos_length_dev, writer=None):
            dev_scores = []
            #loss_list = []
            pos = 0
            gap = 50
            while (pos < len(x_dev)):
                x_batch = x_dev[pos:pos+gap]
                pos_batch = pos_dev[pos:pos+gap]
                y_batch = y_dev[pos:pos+gap]
                sequence_length = length_dev[pos:pos+gap]
                pos_sequence_length = pos_length_dev[pos:pos+gap]
                pos += gap
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_pos: pos_batch,
                    model.input_y: y_batch,
                    model.sequence_length: sequence_length,
                    model.dropout_keep_prob: 1.0
                    }
                #step, loss, scores = sess.run(
                #    [global_step, model.loss, model.prob],
                #    feed_dict)
                step, scores = sess.run(
                    [global_step, model.prob],
                    feed_dict)
                dev_scores = dev_scores + list([s[0] for s in scores])
                #loss_list.append(loss)
            gold_scores = [t[0] for t in y_dev]
            pred_scores = dev_scores[:]
            fpr, tpr, _ = roc_curve(gold_scores, pred_scores, pos_label=1)
            roc_auc = auc(fpr, tpr)
            prec, recall, _ = precision_recall_curve(gold_scores, pred_scores, pos_label=1)
            pr_auc = auc(recall, prec)
            #avg_loss = np.mean(loss_list)
            print("dev roc_auc:", roc_auc,"dev pr_auc:", pr_auc)
            return roc_auc, pr_auc#, avg_loss

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train, pos_train, length_train, pos_length_train, attention_train)), FLAGS.batch_size, FLAGS.num_epochs)
        best_auc = 0.10
        for batch in batches:
            x_batch, y_batch, pos_batch, length_batch, pos_length_batch, attention_batch = zip(*batch)
            train_step(x_batch, pos_batch, y_batch, length_batch, pos_length_batch, attention_batch)
            current_step = tf.train.global_step(sess, global_step)
            if (current_step% FLAGS.evaluate_every == 0):
                print("\n Evaluation:")
                roc_auc, pr_auc = dev_step(x_dev, pos_dev, y_dev, length_dev, pos_length_dev)
                # model selection criteria: roc_auc
                #if (best_auc < roc_auc):
                #    best_auc = roc_auc
                if (best_auc < pr_auc):
                    best_auc = pr_auc
                    print("best pr auc:", best_auc)
                    checkpoint_prefix = os.path.join(checkpoint_dir, "best_model")
                    path = saver.save(sess, checkpoint_prefix)
                    print("Saved best model checkpoint.")




