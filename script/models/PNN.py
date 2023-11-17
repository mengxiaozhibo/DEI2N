#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/1 09:48

from models.base_model import Model
from common.utils import *


class PNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(PNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                  ATTENTION_SIZE,
                                  use_negsampling)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum], 1)

        # Fully connected layer
        self.build_fcn_net(inp, use_dice=False)