#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# @Time    : 2023/1/1 09:51

from common.rnn import dynamic_rnn
from models.base_model import Model
from common.utils import *


class DIN_V2_Gru_Vec_attGru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(DIN_V2_Gru_Vec_attGru, self).__init__(n_uid, n_mid, n_cat,
                                                    EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                    use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)
        print('self.item_eb.get_shape()',self.item_eb.get_shape())
        print('rnn_outputs.get_shape()',rnn_outputs.get_shape())
        print('att_outputs.get_shape())',att_outputs.get_shape())
        print('alphas.get_shape()',alphas.get_shape())
        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        #inp = tf.concat([self.uid_batch_embedded, self.item_eb, final_state2, self.item_his_eb_sum], 1)
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1)
        print('inp.get_shape()',inp.get_shape())
        self.build_fcn_net(inp, use_dice=True)