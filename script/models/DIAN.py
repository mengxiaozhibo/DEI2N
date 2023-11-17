#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/1 09:49

from models.base_model import Model
from common.utils import *


class DIAN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(DIAN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                  ATTENTION_SIZE,
                                  use_negsampling)

        # UIM Attention layer
        is_training = False
        with tf.name_scope('UIM_layer'):
            inp = tf.concat([self.uid_batch_embedded, self.trigger_eb, self.item_his_eb_mean, self.item_his_eb_sum * self.trigger_eb], 1)
            dnn1 = tf.layers.dense(inp, 200, activation=None, name='f1_uim')
            bn1 = tf.contrib.layers.batch_norm(dnn1, is_training=is_training, activation_fn=tf.nn.relu, scope='bn1_uim')
            dnn2 = tf.layers.dense(bn1, EMBEDDING_DIM * 2, activation=None, name='f2_uim')
            bn2 = tf.contrib.layers.batch_norm(dnn2, is_training=is_training, activation_fn=tf.nn.relu, scope='bn2_uim')
            dnn3 = tf.layers.dense(bn2, 1, activation=None, name='f3_uim')
            uim_logit = tf.nn.sigmoid(dnn3)
            self.aux_loss = - tf.reduce_mean(tf.log(uim_logit) * self.target_aux_ph[0] + self.target_aux_ph[1] * tf.log(1 - uim_logit))

        with tf.name_scope('Trigger_Aware_net'):
            Hta_s = multihead_target_attention(queries=self.item_eb,
                                               keys=self.item_his_eb,
                                               num_heads=4,
                                               activation_fn=None,
                                               key_masks=self.mask,
                                               query_masks=None,
                                               num_units=HIDDEN_SIZE,
                                               num_output_units=EMBEDDING_DIM * 2,
                                               name='short_seq_target_attention')
            Hta_s = tf.reduce_sum(Hta_s, 1)
            Hta_l = multihead_target_attention(queries=self.item_eb,
                                               keys=self.item_his_hard_eb,
                                               num_heads=4,
                                               activation_fn=None,
                                               key_masks=self.mask_hard,
                                               query_masks=None,
                                               num_units=HIDDEN_SIZE,
                                               num_output_units=EMBEDDING_DIM * 2,
                                               name='long_seq_target_attention')
            Hta_l = tf.reduce_sum(Hta_l, 1)

            Htri_s = multihead_target_attention(queries=self.trigger_eb,
                                                keys=self.item_his_eb,
                                                num_heads=4,
                                                activation_fn=None,
                                                key_masks=self.mask,
                                                query_masks=None,
                                                num_units=HIDDEN_SIZE,
                                                num_output_units=EMBEDDING_DIM * 2,
                                                name='short_seq_trigger_attention')
            Htri_s = tf.reduce_sum(Htri_s, 1)
            Htri_l = multihead_target_attention(queries=self.trigger_eb,
                                                keys=self.item_his_hard_eb,
                                                num_heads=4,
                                                activation_fn=None,
                                                key_masks=self.mask_hard,
                                                query_masks=None,
                                                num_units=HIDDEN_SIZE,
                                                num_output_units=EMBEDDING_DIM * 2,
                                                name='long_seq_trigger_attention')
            Htri_l = tf.reduce_sum(Htri_l, 1)


            inp = tf.concat(
                [self.uid_batch_embedded, self.item_eb, self.trigger_eb,  self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum, Hta_s, Hta_l, Htri_s, Htri_l], -1)
            prop_aware = self.build_fcn_net_DIAN(inp, use_dice=True, name="layer_aware")

        with tf.name_scope('Trigger_Free_net'):
            Hta_s_free = multihead_target_attention(queries=self.item_eb,
                                                    keys=self.item_his_eb,
                                                    num_heads=4,
                                                    activation_fn=None,
                                                    key_masks=self.mask,
                                                    query_masks=None,
                                                    num_units=HIDDEN_SIZE,
                                                    num_output_units=EMBEDDING_DIM * 2,
                                                    name='short_seq_target_attention_free')
            Hta_s_free = tf.reduce_sum(Hta_s_free, 1)
            Hta_l_free = multihead_target_attention(queries=self.item_eb,
                                                    keys=self.item_his_hard_eb,
                                                    num_heads=4,
                                                    activation_fn=None,
                                                    key_masks=self.mask_hard,
                                                    query_masks=None,
                                                    num_units=HIDDEN_SIZE,
                                                    num_output_units=EMBEDDING_DIM * 2,
                                                    name='long_seq_target_attention_free')
            Hta_l_free = tf.reduce_sum(Hta_l_free, 1)
            inp_free = tf.concat(
                [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum, Hta_l_free, Hta_s_free], -1)
            prop_freee = self.build_fcn_net_DIAN(inp_free, use_dice=True, name="layer_free")

        self.build_fcn_net(inp, use_dice=True)
        self.y_hat = prop_freee * uim_logit + prop_aware * (1 - uim_logit)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('Metrics'):
                # Cross-entropy loss and optimizer initialization
                ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
                self.loss = ctr_loss
                if self.use_negsampling:
                    self.loss += 0.1 * self.aux_loss
                tf.summary.scalar('loss', self.loss)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

                # Accuracy metric
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()
