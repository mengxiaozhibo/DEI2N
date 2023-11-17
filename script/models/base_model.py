#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/1 09:31

from common.utils import *
from common.Dice import dice


class Model(object):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
                 maxlen=30, maxlen_hard=20):
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen], name='mid_his_batch_ph')
            self.cat_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen], name='cat_his_batch_ph')
            self.time_stamp_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen], name='time_stamp_his_batch_ph')

            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.cat_batch_ph = tf.placeholder(tf.int32, [None, ], name='cat_batch_ph')

            self.mid_hard_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen_hard], name='mid_hard_his_batch_ph')
            self.cat_hard_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen_hard], name='cat_hard_his_batch_ph')
            self.time_stamp_hard_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen_hard], name='time_stamp_hard_his_batch_ph')

            self.mask_hard = tf.placeholder(tf.float32, [None, None], name='mask_hard')
            self.seq_hard_len_ph = tf.placeholder(tf.int32, [None], name='seq_hard_len_ph')

            self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')


            self.mid_trigger_his_batch_ph = tf.placeholder(tf.int32, [None, 1], name='mid_trigger_his_batch_ph')
            self.cat_trigger_his_batch_ph = tf.placeholder(tf.int32, [None, 1], name='cat_trigger_his_batch_ph')

            self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
            self.target_aux_ph = tf.placeholder(tf.float32, [None, None], name='target_aux_ph')
            self.lr = tf.placeholder(tf.float64, [])
            self.use_negsampling = use_negsampling
            if use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None],
                                                         name='noclk_mid_batch_ph')  # generate 3 item IDs from negative sampling.
                self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_cat_batch_ph')

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM])
            tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)

            self.mid_trigger_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                                         self.mid_trigger_his_batch_ph)
            self.mid_hard_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                                      self.mid_hard_his_batch_ph)

            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                                           self.noclk_mid_batch_ph)

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM])
            tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)
            self.cat_trigger_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var,
                                                                         self.cat_trigger_his_batch_ph)
            self.cat_hard_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var,
                                                                      self.cat_hard_his_batch_ph)

            if self.use_negsampling:
                self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var,
                                                                           self.noclk_cat_batch_ph)

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)

        self.item_his_hard_eb = tf.concat([self.mid_hard_his_batch_embedded, self.cat_hard_his_batch_embedded], 2)
        self.item_his_hard_eb_sum = tf.reduce_sum(self.item_his_hard_eb, 1)
        self.item_his_eb_mean = tf.reduce_mean(self.item_his_eb, 1)
        self.item_his_hard_eb_mean = tf.reduce_mean(self.item_his_hard_eb, 1)

        self.trigger_eb = tf.reduce_sum(tf.concat([self.mid_trigger_his_batch_embedded, self.cat_trigger_his_batch_embedded], 2), 1)

        self.time_embeddings_var = tf.get_variable("time_embedding_var", [86401, EMBEDDING_DIM])
        self.item_his_time_eb = tf.nn.embedding_lookup(self.time_embeddings_var, self.time_stamp_his_batch_ph)

        self.item_his_time_hard_eb = tf.nn.embedding_lookup(self.time_embeddings_var, self.time_stamp_hard_his_batch_ph)


        self.seq_hard_len_var = tf.get_variable("seq_length_eb", [maxlen_hard+1, 4])
        self.seq_hard_len_eb = tf.nn.embedding_lookup(self.seq_hard_len_var, self.seq_hard_len_ph)

        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat(
                [self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cat_his_batch_embedded[:, :, 0, :]],
                -1)  # 0 means only using the first negative item ID. 3 item IDs are inputed in the line 24.
            self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb,
                                                [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1],
                                                 36])  # cat embedding 18 concate item embedding 18.

            self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cat_his_batch_embedded], -1)
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)

    def build_fcn_net(self, inp, use_dice=False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('Metrics'):
                # Cross-entropy loss and optimizer initialization
                ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
                self.loss = ctr_loss
                if self.use_negsampling:
                    self.loss += self.aux_loss
                tf.summary.scalar('loss', self.loss)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

                # Accuracy metric
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()


    def build_fcn_net2(self, inp, use_dice=False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 300, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 120, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('Metrics'):
                # Cross-entropy loss and optimizer initialization
                ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
                self.loss = ctr_loss
                if self.use_negsampling:
                    self.loss += self.aux_loss
                tf.summary.scalar('loss', self.loss)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

                # Accuracy metric
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

    def build_dnn_net(self, inp, use_dice, is_training=True):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn11', training=is_training)
        dnn = tf.layers.dense(bn1, 18, activation=None, name='IL_f1')
        if use_dice:
            dnn1 = dice(dnn, name='dice_dnn_1')
        else:
            dnn1 = prelu(dnn, 'prelu_dnn_1')
        # dnn2 = tf.layers.dense(dnn1, 18, activation=None, name='IL_f2')
        # bn2 = tf.layers.batch_normalization(inputs=dnn2, name='bn22', training=is_training)
        # if use_dice:
        #     dnn3 = dice(bn2, name='dice_dnn_2')
        # else:
        #     dnn3 = prelu(bn2, 'prelu_dnn_2')
        return dnn1

    def build_fcn_net_DIAN(self, inp, use_dice=False, name="layer"):
        bn1 = tf.layers.batch_normalization(inputs=inp, name=name+'bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name=name+'f1')
        if use_dice:
            dnn1 = dice(dnn1, name=name+'dice_1')
        else:
            dnn1 = prelu(dnn1, name+'prelu1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name=name+'f2')
        if use_dice:
            dnn2 = dice(dnn2, name=name+'dice_2')
        else:
            dnn2 = prelu(dnn2, name+'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name=name + 'f3')
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat

    def build_lr_net(self, inp):
        dnn = tf.layers.dense(inp, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn) + 0.00000001

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('Metrics'):
                # Cross-entropy loss and optimizer initialization
                ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
                self.loss = ctr_loss
                if self.use_negsampling:
                    self.loss += self.aux_loss
                tf.summary.scalar('loss', self.loss)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

                # Accuracy metric
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag=None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_loss_v2(self, h_states, click_seq, noclick_seq, time_stamp_seq, mask, stag=None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq, time_stamp_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq, time_stamp_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat

    def train(self, sess, inps):
        if self.use_negsampling:
            loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer],
                                                   feed_dict={
                                                       self.uid_batch_ph: inps[0],
                                                       self.mid_batch_ph: inps[1],
                                                       self.cat_batch_ph: inps[2],
                                                       self.mid_trigger_his_batch_ph: inps[3],
                                                       self.cat_trigger_his_batch_ph: inps[4],
                                                       self.mid_his_batch_ph: inps[5],
                                                       self.cat_his_batch_ph: inps[6],
                                                       self.time_stamp_his_batch_ph: inps[7],
                                                       self.mask: inps[8],
                                                       self.mid_hard_his_batch_ph: inps[9],
                                                       self.cat_hard_his_batch_ph: inps[10],
                                                       self.time_stamp_hard_his_batch_ph: inps[11],
                                                       self.mask_hard: inps[12],
                                                       self.target_ph: inps[13],
                                                       self.seq_len_ph: inps[14],
                                                       self.seq_hard_len_ph: inps[15],
                                                       self.lr: inps[16],
                                                       self.noclk_mid_batch_ph: inps[17],
                                                       self.noclk_cat_batch_ph: inps[18],
                                                       self.target_aux_ph: inps[19],
                                                   })
            return loss, accuracy, aux_loss
        else:
            loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_trigger_his_batch_ph: inps[3],
                self.cat_trigger_his_batch_ph: inps[4],
                self.mid_his_batch_ph: inps[5],
                self.cat_his_batch_ph: inps[6],
                self.time_stamp_his_batch_ph: inps[7],
                self.mask: inps[8],
                self.mid_hard_his_batch_ph: inps[9],
                self.cat_hard_his_batch_ph: inps[10],
                self.time_stamp_hard_his_batch_ph: inps[11],
                self.mask_hard: inps[12],
                self.target_ph: inps[13],
                self.seq_len_ph: inps[14],
                self.seq_hard_len_ph: inps[15],
                self.lr: inps[16],
                self.target_aux_ph: inps[19],
            })
            # print('self.seq_len_ph',self.seq_len_ph)
            return loss, accuracy, 0

    def calculate(self, sess, inps):
        if self.use_negsampling:
            probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss],
                                                       feed_dict={
                                                           self.uid_batch_ph: inps[0],
                                                           self.mid_batch_ph: inps[1],
                                                           self.cat_batch_ph: inps[2],
                                                           self.mid_trigger_his_batch_ph: inps[3],
                                                           self.cat_trigger_his_batch_ph: inps[4],
                                                           self.mid_his_batch_ph: inps[5],
                                                           self.cat_his_batch_ph: inps[6],
                                                           self.time_stamp_his_batch_ph: inps[7],
                                                           self.mask: inps[8],
                                                           self.mid_hard_his_batch_ph: inps[9],
                                                           self.cat_hard_his_batch_ph: inps[10],
                                                           self.time_stamp_hard_his_batch_ph: inps[11],
                                                           self.mask_hard: inps[12],
                                                           self.target_ph: inps[13],
                                                           self.seq_len_ph: inps[14],
                                                           self.seq_hard_len_ph: inps[15],
                                                           self.noclk_mid_batch_ph: inps[16],
                                                           self.noclk_cat_batch_ph: inps[17],
                                                           self.target_aux_ph: inps[18],
                                                       })
            return probs, loss, accuracy, aux_loss
        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_trigger_his_batch_ph: inps[3],
                self.cat_trigger_his_batch_ph: inps[4],
                self.mid_his_batch_ph: inps[5],
                self.cat_his_batch_ph: inps[6],
                self.time_stamp_his_batch_ph: inps[7],
                self.mask: inps[8],
                self.mid_hard_his_batch_ph: inps[9],
                self.cat_hard_his_batch_ph: inps[10],
                self.time_stamp_hard_his_batch_ph: inps[11],
                self.mask_hard: inps[12],
                self.target_ph: inps[13],
                self.seq_len_ph: inps[14],
                self.seq_hard_len_ph: inps[15],
                self.target_aux_ph: inps[18],
            })
            return probs, loss, accuracy, 0

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)
