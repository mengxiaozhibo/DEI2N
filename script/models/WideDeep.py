#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/1 09:44

from models.base_model import Model
from common.utils import *

class WideDeep(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(WideDeep, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                       ATTENTION_SIZE,
                                       use_negsampling)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb,  self.item_his_eb_sum, self.trigger_eb], 1)
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        dnn1 = prelu(dnn1, 'p1')
        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        dnn2 = prelu(dnn2, 'p2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        d_layer_wide = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_eb * self.item_his_eb_sum], axis=-1)
        d_layer_wide = tf.layers.dense(d_layer_wide, 2, activation=None, name='f_fm')
        self.y_hat = tf.nn.softmax(dnn3 + d_layer_wide) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()


        # inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        # # Fully connected layer
        # bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        # dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        # dnn1 = prelu(dnn1, 'p1')
        # dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        # dnn2 = prelu(dnn2, 'p2')
        # dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        # d_layer_wide = tf.concat([tf.concat([self.item_eb,self.item_his_eb_sum], axis=-1),
        #                         self.item_eb * self.item_his_eb_sum], axis=-1)
        # d_layer_wide = tf.layers.dense(d_layer_wide, 2, activation=None, name='f_fm')
        # self.y_hat = tf.nn.softmax(dnn3 + d_layer_wide)
        #
        # with tf.name_scope('Metrics'):
        #     # Cross-entropy loss and optimizer initialization
        #     self.loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
        #     tf.summary.scalar('loss', self.loss)
        #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        #
        #     # Accuracy metric
        #     self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
        #     tf.summary.scalar('accuracy', self.accuracy)
        # self.merged = tf.summary.merge_all()