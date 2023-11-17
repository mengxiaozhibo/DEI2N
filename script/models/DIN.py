#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/1 09:49

from models.base_model import Model
from common.utils import *


class DIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(DIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                  ATTENTION_SIZE,
                                  use_negsampling)

        # # Attention layer
        # with tf.name_scope('Attention_layer'):
        #     attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask)
        #     att_fea = tf.reduce_sum(attention_output, 1)
        #     tf.summary.histogram('att_fea', att_fea)
        # inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, att_fea], -1)
        # # Fully connected layer
        # self.build_fcn_net(inp, use_dice=True)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask, stag='click')
            att_fea = tf.reduce_sum(attention_output, 1)
            attention_output1 = din_attention(self.trigger_eb, self.item_his_eb, ATTENTION_SIZE, self.mask,
                                              stag='click_hard')
            att_fea1 = tf.reduce_sum(attention_output1, 1)
            tf.summary.histogram('att_fea', att_fea)
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum, att_fea1, att_fea], -1)
        # inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
        #                  self.item_eb * self.item_his_eb_sum, att_fea], -1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)
