#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/1 09:43

from common.rnn import dynamic_rnn
from common.utils import *
from models.base_model import Model


class DIN_V2_Gru_Gru_att(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(DIN_V2_Gru_Gru_att, self).__init__(n_uid, n_mid, n_cat,
                                                 EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                 use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_outputs', rnn_outputs2)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs2, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            att_fea = tf.reduce_sum(att_outputs, 1)
            tf.summary.histogram('att_fea', att_fea)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, att_fea], 1)
        self.build_fcn_net(inp, use_dice=True)