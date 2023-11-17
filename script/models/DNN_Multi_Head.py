#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/1 09:52

from models.base_model import Model
from common.utils import *


class DNN_Multi_Head(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True, maxlen=30):
        super(DNN_Multi_Head, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                             ATTENTION_SIZE,
                                             use_negsampling, maxlen)
        # print('self.item_eb.get_shape()', self.item_eb.get_shape())
        # print('self.item_his_eb.get_shape()', self.item_his_eb.get_shape())
        # other_embedding_size = 2
        # self.position_his = tf.range(maxlen)
        # self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, other_embedding_size])
        # self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
        # self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.item_his_eb)[0], 1])  # B*T,E
        # self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.item_his_eb)[0], -1,
        #                                                          self.position_his_eb.get_shape().as_list()[
        #                                                              1]])  # B,T,E
        # with tf.name_scope("multi_head_attention"):
        #     multihead_attention_outputs = self_multi_head_attn(self.item_his_eb, num_units=EMBEDDING_DIM * 2,
        #                                                        num_heads=4, dropout_rate=0, is_training=True)
        #     print('multihead_attention_outputs.get_shape()', multihead_attention_outputs.get_shape())
        #     multihead_attention_outputs1 = tf.compat.v1.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4,
        #                                                              activation=tf.nn.relu)
        #     multihead_attention_outputs1 = tf.compat.v1.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
        #     multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
        #     # multihead_attention_outputs = layer_norm(multihead_attention_outputs, name='multi_head_attention')
        # aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
        #                                  self.noclk_item_his_eb[:, 1:, :],
        #                                  self.mask[:, 1:], stag="gru")
        # self.aux_loss = aux_loss_1
        #
        # inp = tf.concat(
        #     [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum], 1)
        # with tf.name_scope("multi_head_attention"):
        #     multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36,
        #                                                            num_heads=4, dropout_rate=0, is_training=True)
        #     for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
        #         multihead_attention_outputs3 = tf.compat.v1.layers.dense(multihead_attention_outputs_v2,
        #                                                                  EMBEDDING_DIM * 4, activation=tf.nn.relu)
        #         multihead_attention_outputs3 = tf.compat.v1.layers.dense(multihead_attention_outputs3,
        #                                                                  EMBEDDING_DIM * 2)
        #         multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
        #         # multihead_attention_outputs_v2= layer_norm(multihead_attention_outputs_v2, name='multi_head_attention'+str(i))
        #         print('multihead_attention_outputs_v2.get_shape()', multihead_attention_outputs_v2.get_shape())
        #         with tf.name_scope('Attention_layer' + str(i)):
        #             # 这里使用position embedding来算attention
        #             print('self.position_his_eb.get_shape()', self.position_his_eb.get_shape())
        #             print('self.item_eb.get_shape()', self.item_eb.get_shape())
        #             attention_output, attention_score, attention_scores_no_softmax = din_attention_new(self.item_eb,
        #                                                                                                multihead_attention_outputs_v2,
        #                                                                                                self.position_his_eb,
        #                                                                                                ATTENTION_SIZE,
        #                                                                                                self.mask,
        #                                                                                                stag=str(i))
        #             print('attention_output.get_shape()', attention_output.get_shape())
        #             att_fea = tf.reduce_sum(attention_output, 1)
        #             inp = tf.concat([inp, att_fea], 1)
        # # Fully connected layer
        # self.build_fcn_net(inp, use_dice=True)

#####add trigger attention

        other_embedding_size = 2
        self.position_his = tf.range(maxlen)
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, other_embedding_size])
        self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
        self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.item_his_eb)[0], 1])  # B*T,E
        self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.item_his_eb)[0], -1,
                                                                 self.position_his_eb.get_shape().as_list()[
                                                                     1]])  # B,T,E
        with tf.name_scope("multi_head_attention"):
            multihead_attention_outputs = self_multi_head_attn(self.item_his_eb, num_units=EMBEDDING_DIM * 2,
                                                               num_heads=4, dropout_rate=0, is_training=True)
            print('multihead_attention_outputs.get_shape()', multihead_attention_outputs.get_shape())
            multihead_attention_outputs1 = tf.compat.v1.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4,
                                                                     activation=tf.nn.relu)
            multihead_attention_outputs1 = tf.compat.v1.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
            multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            # multihead_attention_outputs = layer_norm(multihead_attention_outputs, name='multi_head_attention')
        aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                         self.noclk_item_his_eb[:, 1:, :],
                                         self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum], 1)
        with tf.name_scope("multi_head_attention"):
            multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36,
                                                                   num_heads=4, dropout_rate=0, is_training=True)
            for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                multihead_attention_outputs3 = tf.compat.v1.layers.dense(multihead_attention_outputs_v2,
                                                                         EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs3 = tf.compat.v1.layers.dense(multihead_attention_outputs3,
                                                                         EMBEDDING_DIM * 2)
                multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                # multihead_attention_outputs_v2= layer_norm(multihead_attention_outputs_v2, name='multi_head_attention'+str(i))
                print('multihead_attention_outputs_v2.get_shape()', multihead_attention_outputs_v2.get_shape())
                with tf.name_scope('Attention_layer' + str(i)):
                    # 这里使用position embedding来算attention
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_new(self.item_eb,
                                                                                                       multihead_attention_outputs_v2,
                                                                                                       self.position_his_eb,
                                                                                                       ATTENTION_SIZE,
                                                                                                       self.mask,
                                                                                                       stag=str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)

                    attention_output_tr, attention_score_tr, attention_scores_no_softmax_tr = din_attention_new(self.trigger_eb,
                                                                                                       multihead_attention_outputs_v2,
                                                                                                       self.position_his_eb,
                                                                                                       ATTENTION_SIZE,
                                                                                                       self.mask,
                                                                                                       stag='tr'+str(i))
                    att_fea_tr = tf.reduce_sum(attention_output_tr, 1)
                    inp = tf.concat([inp, att_fea, att_fea_tr], 1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)
