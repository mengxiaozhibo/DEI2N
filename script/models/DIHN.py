#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/1 09:49

from models.base_model import Model
from common.utils import *


class DIHN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(DIHN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                  ATTENTION_SIZE,
                                  use_negsampling)

        # UIM Attention layer
        is_training = True
        with tf.name_scope('UIM_layer'):
            attention_output = din_attention_hian(self.trigger_eb, self.item_his_eb, self.time_stamp_his_batch_ph,
                                                  ATTENTION_SIZE, self.mask, stag='click_uim')
            att_fea = tf.reduce_sum(attention_output, 1)
            inp = tf.concat([self.uid_batch_embedded, self.trigger_eb, att_fea, self.item_his_eb_sum,
                             self.item_his_eb_sum * self.trigger_eb], 1)
            dnn1 = tf.layers.dense(inp, 200, activation=None, name='f1_uim')
            bn1 = tf.contrib.layers.batch_norm(dnn1, is_training=is_training, activation_fn=tf.nn.relu, scope='bn1_uim')

            dnn2 = tf.layers.dense(bn1, EMBEDDING_DIM * 2, activation=None, name='f2_uim')
            bn2 = tf.contrib.layers.batch_norm(dnn2, is_training=is_training, activation_fn=tf.nn.relu, scope='bn2_uim')
            dnn3 = tf.layers.dense(bn2, 2, activation=None, name='f3_uim')
            uim_logit = tf.nn.softmax(dnn3) + 0.00000001
            self.aux_loss = - tf.reduce_mean(tf.log(uim_logit) * self.target_aux_ph)

            fusing_embedding = tf.multiply(dnn2, self.trigger_eb) + tf.multiply(1 - dnn2, self.item_eb)
            # fusing_embedding = tf.concat([self.trigger_eb,self.item_eb],1)

            clicks_trans_block = SelfAttentionPooling(
                num_heads=2,
                key_mask=self.mask,
                query_mask=self.mask,
                length=30,
                linear_key_dim=HIDDEN_SIZE,
                linear_value_dim=HIDDEN_SIZE,
                output_dim=EMBEDDING_DIM * 2,
                hidden_dim=EMBEDDING_DIM * 4,
                num_layer=2,
                keep_prob=1.0
            )
            clicks_trans_output = clicks_trans_block.build(self.item_his_eb, reuse=False,
                                                           scope='clicks_trans')  # (batch_size, 30, output_dim)

        with tf.name_scope('hybrid_interest_extracting_module'):
            hard_trans_block = SelfAttentionPooling(
                num_heads=2,
                key_mask=self.mask_hard,
                query_mask=self.mask_hard,
                length=20,
                linear_key_dim=HIDDEN_SIZE,
                linear_value_dim=HIDDEN_SIZE,
                output_dim=EMBEDDING_DIM * 2,
                hidden_dim=EMBEDDING_DIM * 4,
                num_layer=2,
                keep_prob=1.0
            )
            hard_trans_output = hard_trans_block.build(self.item_his_hard_eb, reuse=False,
                                                       scope='hard_trans')  # (batch_size, 30, output_dim)
            subcate_clicks_hard_pool_res = tf.reduce_mean(hard_trans_output, axis=1)

            # att_fea_fusing = time_attention_pooling(clicks_trans_output, fusing_embedding,
            #                                             self.mask, False, 'click_attention_pooling')
            att_fea_fusing = din_attention_hian(fusing_embedding, clicks_trans_output, self.time_stamp_his_batch_ph,
                                                    ATTENTION_SIZE, self.mask, stag='fusing_gate_attention')
            att_fea_fusing =tf.reduce_sum(att_fea_fusing,1)

        inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_eb * self.item_his_eb_sum, self.item_his_eb_sum,
             self.trigger_eb, att_fea_fusing, subcate_clicks_hard_pool_res], -1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)