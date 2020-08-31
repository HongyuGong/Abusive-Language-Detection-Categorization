"""
Test abusive language categorization model
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
from data_util import data_helpers, tag_data_helpers, category_data_helpers, eval_helpers
from model.abuse_categorizer import AbuseCategorizer

tf.set_random_seed(111)
# multitask setting
tf.flags.DEFINE_boolean("multitask", True, "use multitasking for model training")
tf.flags.DEFINE_integer("primary_label", 0, "primary label in multitasking (0: gender, 1: race, 2: race, 3: ideology")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("pos_vocab_size", 26, "Vocab size of POS tags")
tf.flags.DEFINE_integer("pos_embedding_dim", 25, "Dcimensionality of pos tag embedding (default: 20)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
#tf.flags.DEFINE_boolean("use_attention", True, "whether to add supervised attention")
tf.flags.DEFINE_float("attention_lambda", 0.2, "Supervised attention lambda (default: 0.05)")
tf.flags.DEFINE_string("attention_loss_type", 'encoded', "loss function of attention")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularizaion lambda (default: 0.05)")
tf.flags.DEFINE_integer("hidden_size", 300, "Dimensionality of RNN cell (default: 300)")
tf.flags.DEFINE_integer("pos_hidden_size", 25, "Dimensionality of POS-RNN cell")
tf.flags.DEFINE_integer("attention_size", 20, "Dimensionality of attention scheme (default: 50)")
tf.flags.DEFINE_boolean("use_pos_flag", True, "use the sequence of POS tags")
# Training parameters -- evaluate_every:100
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 400, "Number of training epochs (default: 200)")
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

def scoreUtil(init_embed, x_dev, length_dev, pos_dev, pos_length_dev, model_path):
    with tf.Graph().as_default():
      session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
      sess = tf.Session(config=session_conf)
      with sess.as_default():
        model = AbuseCategorizer(
          max_sequence_length=param.max_sent_len,
          num_classes=2,
          pos_vocab_size = FLAGS.pos_vocab_size,
          init_embed=init_embed,
          hidden_size=FLAGS.hidden_size,
          attention_size=FLAGS.attention_size,
          keep_prob=FLAGS.dropout_keep_prob,
          attention_lambda = FLAGS.attention_lambda,
          attention_loss_type = FLAGS.attention_loss_type,
          l2_reg_lambda=FLAGS.l2_reg_lambda,
          use_pos_flag = FLAGS.use_pos_flag,
          category_weights = category_weights)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        saver = tf.train.Saver(tf.all_variables())
        # Initialize all variables
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, model_path)

        dev_gender_scores = []
        dev_race_scores = []
        dev_appear_scores = []
        dev_idea_scores = []
        pos = 0
        gap = 50
        while (pos < len(x_dev)):
          x_batch = x_dev[pos:pos+gap]
          pos_batch = pos_dev[pos:pos+gap]
          length_batch = length_dev[pos:pos+gap]
          pos_length_batch = pos_length_dev[pos:pos+gap]
          #y_batch = y_dev[pos:pos+gap]
          pos += gap
          # score sentences
          feed_dict = {
              model.input_x: x_batch,
              model.input_pos: pos_batch,
              model.sequence_length: length_batch,
              model.dropout_keep_prob: 1.0
              }
          #step, scores = sess.run([global_step, model.prob], feed_dict)
          step, scores_gender, scores_race, scores_appear, scores_idea = sess.run(
              [global_step, model.prob_gender, model.prob_race, model.prob_appear, model.prob_idea],
              feed_dict)
          dev_gender_scores = dev_gender_scores + [s[0] for s in scores_gender]
          dev_race_scores = dev_race_scores + [s[0] for s in scores_race]
          dev_appear_scores = dev_appear_scores + [s[0] for s in scores_appear]
          dev_idea_scores = dev_idea_scores + [s[0] for s in scores_idea]
    return dev_gender_scores, dev_race_scores, dev_appear_scores, dev_idea_scores


def scoreComments(model_path, data_type):
    """
    Score comments with saved model
    """
    vocabulary, pos_vocabulary, init_embed = data_helpers.loadVocabEmb()
    x, length, attention, pos, pos_length, y_gender, \
       y_race, y_appear, y_idea = category_data_helpers.loadCategoryData(data_type=data_type)
    gold_gender_scores = [s[0] for s in y_gender]
    gold_race_scores = [s[0] for s in y_race]
    gold_appear_scores = [s[0] for s in y_appear]
    gold_idea_scores = [s[0] for s in y_idea]
    
    pred_gender_scores, pred_race_scores, pred_appear_scores, pred_idea_scores = \
                       scoreUtil(init_embed, x, length, pos, pos_length, model_path)
    
    return (gold_gender_scores, gold_race_scores, gold_appear_scores, gold_idea_scores), \
           (pred_gender_scores, pred_race_scores, pred_appear_scores, pred_idea_scores)


if __name__=="__main__":
    # locate checkpoint
    if FLAGS.checkpoint == "":
        out_dir = os.path.abspath(os.path.join(os.path.pardir, "model"))
    else:
        out_dir = FLAGS.checkpoint
    if (FLAGS.multitask):
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "multitask_attn="+FLAGS.attention_loss_type+"_lambda="+str(FLAGS.attention_lambda)))
    else:
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "singletask_attn="+FLAGS.attention_loss_type+"_lambda="+str(FLAGS.attention_lambda)))
    model_path = os.path.join(checkpoint_dir, "best_model")

    # evaluate on train data
    train_gold_scores, train_pred_scores = scoreComments(model_path, data_type="train")
    # evaluate on test data
    test_gold_scores, test_pred_scores = scoreComments(model_path, data_type="test")

    tasks = ["gender", "race", "appear", "idea"]
    for task, train_task_gold_scores, train_task_pred_scores, test_task_gold_scores, test_task_pred_scores \
        in zip(tasks, train_gold_scores, train_pred_scores, test_gold_scores, test_pred_scores):
        print("Evaluate task: {}".format(task))
        # roc auc
        eval_helpers.evalROC(test_task_gold_scores, test_task_pred_scores)
        # pr auc
        eval_helpers.evalPR(test_task_gold_scores, test_task_pred_scores)
        # f1 score
        eval_helpers.evalFscore(train_task_gold_scores, train_task_pred_scores,
                                test_task_gold_scores, test_task_pred_scores)   




