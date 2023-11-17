#!/usr/bin/env python
# /*
# MIT License
#
# Copyright (c) 2020 skx300
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -*- coding: utf-8 -*-
# @Time    : 2023/1/1 09:49
from models.base_model import Model
from common.utils import *


class DEI2N_MHTA(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(DEI2N_MHTA, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                  ATTENTION_SIZE,
                                  use_negsampling)

        is_training = True

        with tf.name_scope('UIM_layer'):
            inp = tf.concat([self.uid_batch_embedded, self.item_his_hard_eb_sum, self.trigger_eb,
                             self.trigger_eb * self.item_his_hard_eb_sum,
                             self.seq_hard_len_eb], 1)
            dnn1 = tf.layers.batch_normalization(inputs=inp, name='bn1_uin', training=is_training)
            dnn1 = tf.layers.dense(dnn1, EMBEDDING_DIM , activation=tf.nn.relu, name='f1_uim')
            # dnn2 = tf.layers.batch_normalization(inputs=dnn1, name='bn2_uin', training=is_training)
            # dnn3 = tf.layers.dense(dnn2, EMBEDDING_DIM, activation=tf.nn.relu, name='f2_uim')
            # dnn3 = tf.layers.batch_normalization(inputs=dnn3, name='bn3_uin', training=is_training)
            dnn3 = tf.layers.dense(dnn1, 1, activation=tf.sigmoid, name='f3_uim')

        with tf.name_scope('Attention_layer'):
            # clicks_trans_block = SelfAttentionPooling(
            #     num_heads=2,
            #     key_mask=self.mask,
            #     query_mask=self.mask,
            #     length=30,
            #     linear_key_dim=HIDDEN_SIZE,
            #     linear_value_dim=HIDDEN_SIZE,
            #     output_dim=EMBEDDING_DIM * 2,
            #     hidden_dim=EMBEDDING_DIM * 4,
            #     num_layer=1,
            #     keep_prob=self.keep_prob
            # )
            item_his_eb_mix = tf.concat([self.item_his_eb, self.item_his_time_eb], axis=2)
            item_his_eb_mix = tf.layers.dense(item_his_eb_mix, EMBEDDING_DIM * 2, activation=None,
                                              name='item_his_eb_mix')
            # clicks_trans_output = clicks_trans_block.build(item_his_eb_mix, reuse=False,
            #                                                scope='clicks_trans')  # (batch_size, 30, output_dim)

            attention_output_soft = multihead_target_attention(queries=self.item_eb,
                                                               keys=item_his_eb_mix,
                                                               num_heads=2,
                                                               activation_fn=None,
                                                               key_masks=self.mask,
                                                               query_masks=None,
                                                               num_units=HIDDEN_SIZE,
                                                               num_output_units=EMBEDDING_DIM * 2,
                                                               name='soft_seq_target_attention')

            att_fea_soft = tf.reduce_sum(attention_output_soft, 1)
            attention_output_trigger = multihead_target_attention(queries=self.trigger_eb,
                                                                  keys=item_his_eb_mix,
                                                                  num_heads=2,
                                                                  activation_fn=None,
                                                                  key_masks=self.mask,
                                                                  query_masks=None,
                                                                  num_units=HIDDEN_SIZE,
                                                                  num_output_units=EMBEDDING_DIM * 2,
                                                                  name='soft_seq_trigger_attention')
            att_fea_trigger = tf.reduce_sum(attention_output_trigger, 1)

            attention_output_hard = multihead_target_attention(queries=self.item_eb,
                                                                  keys=self.item_his_hard_eb,
                                                                  num_heads=2,
                                                                  activation_fn=None,
                                                                  key_masks=self.mask_hard,
                                                                  query_masks=None,
                                                                  num_units=HIDDEN_SIZE,
                                                                  num_output_units=EMBEDDING_DIM * 2,
                                                                  name='hard_seq_target_attention')
            att_fea_hard = tf.reduce_sum(attention_output_hard, 1)

            tf.summary.histogram('att_fea_soft', att_fea_soft)

            att_fea_mix = tf.multiply(att_fea_trigger, dnn3) + tf.multiply(att_fea_soft, 1 - dnn3)

        # with tf.name_scope('trigger_target_interaction_layer'):
            Hadamard_fea = tf.multiply(self.trigger_eb, self.item_eb)
            cross_inp = tf.concat(
                [self.trigger_eb, self.item_eb, self.trigger_eb-self.item_eb ,Hadamard_fea], 1)

        inp = tf.concat(
            [self.uid_batch_embedded, self.item_his_eb_sum, cross_inp,
             self.item_eb * self.item_his_eb_sum, att_fea_hard, att_fea_mix], -1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)