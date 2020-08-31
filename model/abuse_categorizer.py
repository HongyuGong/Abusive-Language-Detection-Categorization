"""
Abusive language categorizer with supervised attention
"""
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from attention import attention


class AbuseCategorizer(object):
    def __init__(self, max_sequence_length, num_classes, pos_vocab_size, init_embed, \
                 hidden_size, attention_size, keep_prob, attention_lambda, attention_loss_type, \
                 l2_reg_lambda, use_pos_flag, category_weights, rnn_cell="lstm"):
        # word index
        self.input_x = tf.placeholder(tf.int32, [None, max_sequence_length], name="input_x")
        # pos index
        self.input_pos = tf.placeholder(tf.int32, [None, max_sequence_length], name="input_pos")
        # sequence length of words
        self.sequence_length = tf.placeholder(tf.int32, [None], name="length")
        # attention over x
        self.input_attention = tf.placeholder(tf.float32, [None, max_sequence_length], name="input_attention")
        self.gender_weight, self.race_weight, self.appear_weight, self.idea_weight = category_weights
        # gender
        self.input_y_gender = tf.placeholder(tf.float32, [None, 2], name="input_y_gender")
        # race
        self.input_y_race = tf.placeholder(tf.float32, [None, 2], name="input_y_race")
        # appearance
        self.input_y_appear = tf.placeholder(tf.float32, [None, 2], name="input_y_appear")
        # ideology
        self.input_y_idea = tf.placeholder(tf.float32, [None, 2], name="input_y_idea")
        
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)

        # embedding layer with initialization of words and pos tags
        with tf.name_scope("embedding"):
            W = tf.Variable(init_embed, name="W", dtype=tf.float32)
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_input = self.embedded_chars

        if (use_pos_flag):
            with tf.name_scope("pos_embedding"):
                W_pos = tf.Variable(tf.eye(pos_vocab_size), name="W_pos", dtype=tf.float32)
                self.embedded_pos = tf.nn.embedding_lookup(W_pos, self.input_pos)
                self.embedded_input = tf.concat([self.embedded_chars, self.embedded_pos], axis=-1)
                
        # RNN layer + attention for words
        with tf.variable_scope("bi-rnn"):
            if rnn_cell == "gru":
                rnn_outputs, _ = bi_rnn(GRUCell(hidden_size), GRUCell(hidden_size),\
                                        inputs=self.embedded_input, sequence_length=self.sequence_length, \
                                        dtype=tf.float32)
            elif rnn_cell == "lstm":
                rnn_outputs, _ = bi_rnn(LSTMCell(hidden_size), LSTMCell(hidden_size),\
                                        inputs=self.embedded_input, sequence_length=self.sequence_length, \
                                        dtype=tf.float32)
            else:
                raise Exception(f"Cell tyle {rnn_cell} is not supported!")
            attention_outputs, self.alphas = attention(rnn_outputs, attention_size, return_alphas=True)
            drop_outputs = tf.nn.dropout(attention_outputs, keep_prob)

        # Fully connected layer
        with tf.name_scope("fc-layer-1"):
            fc_dim = 10
            W = tf.Variable(tf.truncated_normal([drop_outputs.get_shape()[1].value, fc_dim], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[fc_dim]), name="b")
            fc_outputs = tf.nn.tanh(tf.nn.xw_plus_b(drop_outputs, W, b))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

        with tf.name_scope("fc-layer-2"):
            # gender
            W_gender = tf.Variable(tf.truncated_normal([fc_outputs.get_shape()[1].value, num_classes], stddev=0.1), name="W_gender")
            b_gender = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_gender")
            self.logits_gender = tf.nn.xw_plus_b(fc_outputs, W_gender, b_gender)
            self.prob_gender = tf.nn.softmax(self.logits_gender)
            l2_loss += tf.nn.l2_loss(W_gender)
            l2_loss += tf.nn.l2_loss(b_gender)
            # race
            W_race = tf.Variable(tf.truncated_normal([fc_outputs.get_shape()[1].value, num_classes], stddev=0.1), name="W_race")
            b_race = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_race")
            self.logits_race = tf.nn.xw_plus_b(fc_outputs, W_race, b_race)
            self.prob_race = tf.nn.softmax(self.logits_race)
            l2_loss += tf.nn.l2_loss(W_race)
            l2_loss += tf.nn.l2_loss(b_race)
            # appearance
            W_appear = tf.Variable(tf.truncated_normal([fc_outputs.get_shape()[1].value, num_classes], stddev=0.1), name="W_appear")
            b_appear = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_appear")
            self.logits_appear = tf.nn.xw_plus_b(fc_outputs, W_appear, b_appear)
            self.prob_appear = tf.nn.softmax(self.logits_appear)
            l2_loss += tf.nn.l2_loss(W_appear)
            l2_loss += tf.nn.l2_loss(b_appear)
            # ideology
            W_idea = tf.Variable(tf.truncated_normal([fc_outputs.get_shape()[1].value, num_classes], stddev=0.1), name="W_idea")
            b_idea = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_idea")
            self.logits_idea = tf.nn.xw_plus_b(fc_outputs, W_idea, b_idea)
            self.prob_idea = tf.nn.softmax(self.logits_idea)
            l2_loss += tf.nn.l2_loss(W_idea)
            l2_loss += tf.nn.l2_loss(b_idea)
            
        with tf.name_scope("cross_entropy"):
            loss_gender = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y_gender, logits=self.logits_gender)
            loss_race = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y_race, logits=self.logits_race)
            loss_appear = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y_appear, logits=self.logits_appear)
            loss_idea = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y_idea, logits=self.logits_idea)
            # weighted entropy loss
            entropy_loss = tf.reduce_mean(self.gender_weight * loss_gender + self.race_weight * loss_race \
                                          + self.appear_weight * loss_appear + self.idea_weight * loss_idea)

            if (attention_loss_type == "encoded"):
                print("Supervised attention with encoded loss.")
                att_shared_dim = 20
                # rationale input_attention: (batch_size, max_sequence_length)
                # W: (max_sequence_length, att_shared_dim)
                # b: (att_shared_dim,)
                # proj: (batch_size, att_shared_dim)
                ration_W = tf.Variable(tf.truncated_normal([self.input_attention.get_shape()[1].value, att_shared_dim], stddev=0.1), name="ration_W")
                ration_b = tf.Variable(tf.constant(0.05, shape=[att_shared_dim]), name="ration_b")
                proj_ration = tf.nn.tanh(tf.nn.xw_plus_b(self.input_attention, ration_W, ration_b))
                alpha_W = tf.Variable(tf.truncated_normal([self.alphas.get_shape()[1].value, att_shared_dim], stddev=0.1), name="alpha_W")
                alpha_b = tf.Variable(tf.constant(0.05, shape=[att_shared_dim]), name="alpha_b")
                proj_alphas = tf.nn.tanh(tf.nn.xw_plus_b(self.alphas, alpha_W, alpha_b))
                # negative of inner product
                attention_loss = -1 * tf.reduce_mean(tf.multiply(proj_ration, proj_alphas))
            elif (attention_loss_type == "l1"):
                print("Supervised attention with L1 loss.")
                attention_loss = tf.reduce_mean(tf.abs(tf.subtract(tf.nn.softmax(self.input_attention), self.alphas)))
            elif (attention_loss_type == "l2"):
                print("Supervised attention with L2 loss.")
                attention_loss = tf.reduce_mean(tf.square(tf.subtract(tf.nn.softmax(self.input_attention), self.alphas)))
            else:
                print("No supervised attention.")
                attention_loss = tf.constant(0.0)
            self.loss = entropy_loss + attention_lambda * attention_loss + l2_reg_lambda * l2_loss
