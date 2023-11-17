#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/12/30 16:29

import numpy
import time
import os
import random
import sys

from dataset_process.data_iterator import DataIterator
from models.DIN import DIN
from models.DIN_V2_Gru_Gru_att import DIN_V2_Gru_Gru_att
from models.DIN_V2_Gru_QA_attGru import DIN_V2_Gru_QA_attGru
from models.DIN_V2_Gru_Vec_attGru import DIN_V2_Gru_Vec_attGru
from models.DIN_V2_Gru_Vec_attGru_Neg import DIN_V2_Gru_Vec_attGru_Neg
from models.DIN_V2_Gru_att_Gru import DIN_V2_Gru_att_Gru
from models.DNN import DNN
from models.DNN_Multi_Head import DNN_Multi_Head
from models.PNN import PNN
from models.DEI2N import DEI2N
from models.DEI2N_NO_IL import DEI2N_NO_IL
from models.DEI2N_NO_UIM import DEI2N_NO_UIM
from models.DEI2N_NO_TIM import DEI2N_NO_TIM
from models.DIHN import DIHN
from models.DEI2N_MHTA import DEI2N_MHTA
from models.WideDeep import WideDeep
from models.DIAN import DIAN
from models.LR import LR
from dataset_process.prepare_data import prepare_data, prepare_data_tgin
from common.utils import *
from settings import *

best_auc = 0.0


def model_selection(model_type, n_uid, n_mid, n_cat):
    print("Model_type: {}".format(model_type))

    if model_type == 'DNN':
        model = DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'PNN':
        model = PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'Wide':
        model = WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DIN':
        model = DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DIN-V2-gru-att-gru':
        model = DIN_V2_Gru_att_Gru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DIN-V2-gru-gru-att':
        model = DIN_V2_Gru_Gru_att(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DIN-V2-gru-qa-attGru':
        model = DIN_V2_Gru_QA_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DIN-V2-gru-vec-attGru':
        model = DIN_V2_Gru_Vec_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DIEN':
        model = DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DMIN':
        model = DNN_Multi_Head(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DEI2N':
        model = DEI2N(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DEI2N_NO_IL':
        model = DEI2N_NO_IL(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DEI2N_NO_UIM':
        model = DEI2N_NO_UIM(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DEI2N_NO_TIM':
        model = DEI2N_NO_TIM(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DIHN':
        model = DIHN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DIAN':
        model = DIAN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'LR':
        model = LR(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif model_type == 'DEI2N_MHTA':
        model = DEI2N_MHTA(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    else:
        print("Invalid model_type : %s", model_type)

    return model


def eval(sess, test_data, model, model_path, maxlen=30, maxlen_hard=20, model_type='DNN'):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    # for src, tri0_src, tri1_src, tgt in test_data:
    for one_pass_data in test_data:
        nums += 1
        src, tgt, tgt_aux = one_pass_data
        uids, mids, cats, time_stamp, trigger_mid, trigger_cat, mid_his, cat_his, time_stamp_his, mid_mask, mid_hard_his, cat_hard_his, time_stamp_hard_his, mid_hard_mask, target, sl, sl_hard, noclk_mids, noclk_cats, target_aux = prepare_data(src,
                                                                                                        tgt,
                                                                                                        tgt_aux,
                                                                                                        maxlen,
                                                                                                        maxlen_hard,
                                                                                                        return_neg=True)
        prob, loss, acc, aux_loss = model.calculate(sess, [uids, mids, cats, trigger_mid, trigger_cat, mid_his, cat_his, time_stamp_his, mid_mask, mid_hard_his, cat_hard_his, time_stamp_hard_his, mid_hard_mask, target, sl, sl_hard,
                                                           noclk_mids, noclk_cats, target_aux])
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        model.save(sess, model_path)
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum


def train(seed=1234, model_type='DNN'):
    if dataset == "electronics":
        train_file = os.path.join(data_path, dataset, "local_train_splitByUser")
        test_file = os.path.join(data_path, dataset, "local_test_splitByUser")
        valid_file = os.path.join(data_path, dataset, "local_valid_splitByUser")
        uid_voc = os.path.join(data_path, dataset, "uid_voc.pkl")
        mid_voc = os.path.join(data_path, dataset, "mid_voc.pkl")
        cat_voc = os.path.join(data_path, dataset, "cat_voc.pkl")
        item_info = os.path.join(data_path, dataset, "item-info")
        reviews_info = os.path.join(data_path, dataset, "reviews-info")
    elif dataset == "content_wise":
        train_file = os.path.join(data_path, dataset, "local_train_splitByUser_cw")
        test_file = os.path.join(data_path, dataset, "local_test_splitByUser_cw")
        valid_file = os.path.join(data_path, dataset, "local_test_splitByUser_cw")
        uid_voc = os.path.join(data_path, dataset, "uid_voc_cw.pkl")
        mid_voc = os.path.join(data_path, dataset, "itemid_voc_cw.pkl")
        cat_voc = os.path.join(data_path, dataset, "series_voc_cw.pkl")
        item_info = os.path.join(data_path, dataset, "item-info_cw")
        reviews_info = os.path.join(data_path, dataset, "user_log_info")
    else:
        print("Please provide valid dataset!")

    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        t1 = time.time()
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info,
                                  batch_size, maxlen, maxlen_hard, shuffle_each_epoch=False)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info,
                                 batch_size, maxlen, maxlen_hard)
        valid_data = DataIterator(valid_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info,
                                  batch_size, maxlen, maxlen_hard)
        print('# Load data time (s):', round(time.time() - t1, 2))

        n_uid, n_mid, n_cat = train_data.get_n()
        print('n_uid: {}, n_mid: {}, n_cat: {}'.format(n_uid, n_mid, n_cat))
        t1 = time.time()
        model = model_selection(model_type, n_uid, n_mid, n_cat)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sys.stdout.flush()
        print('# Contruct model time (s):', round(time.time() - t1, 2))

        t1 = time.time()
        print('test_auc: %.4f -- test_loss: %.4f -- test_accuracy: %.4f -- test_aux_loss: %.4f' % eval(
            sess, test_data, model, best_model_path, maxlen, maxlen_hard, model_type))
        print('# Eval model time (s):', round(time.time() - t1, 2))
        sys.stdout.flush()

        start_time = time.time()
        iter = 0
        lr = 0.001
        print("lr: {}".format(lr))
        for itr in range(2):
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            # for src, tri0_src, tri1_src, tgt in train_data:
            for one_pass_data in train_data:
                src, tgt, tgt_aux = one_pass_data
                uids, mids, cats, time_stamp, trigger_mid, trigger_cat, mid_his, cat_his, time_stamp_his, mid_mask, \
                mid_hard_his, cat_hard_his, time_stamp_hard_his, mid_hard_mask, target, sl, sl_hard, noclk_mids, noclk_cats, target_aux = prepare_data(src, tgt, tgt_aux, maxlen, maxlen_hard, return_neg=True)

                loss, acc, aux_loss = model.train(sess,
                                                  [uids, mids, cats, trigger_mid, trigger_cat, mid_his, cat_his, time_stamp_his, mid_mask, mid_hard_his, cat_hard_his, time_stamp_hard_his, mid_hard_mask, target, sl, sl_hard, lr,
                                                   noclk_mids, noclk_cats, target_aux])
                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                iter += 1

                # Print & Save
                sys.stdout.flush()
                if (iter % test_iter) == 0 and itr>0 and iter>15100:
                    print('[Time] ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    print('Best_auc:', best_auc)
                    print('itr: %d --> iter: %d --> train_loss: %.4f -- train_accuracy: %.4f -- tran_aux_loss: %.4f' % \
                          (itr, iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))
                    print('test_auc: %.4f -- test_loss: %.4f -- test_accuracy: %.4f -- test_aux_loss: %.4f' % eval(
                        sess, valid_data, model, best_model_path, maxlen, maxlen_hard, model_type))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                # if (iter % save_iter) == 0:
                #     print('save model iter: %d' % (iter))
                #     model.save(sess, model_path + "--" + str(iter))


def test(seed=1234, model_type='DNN'):
    if dataset == "electronics":
        train_file = os.path.join(data_path, dataset, "local_train_splitByUser")
        test_file = os.path.join(data_path, dataset, "local_test_splitByUser")
        valid_file = os.path.join(data_path, dataset, "local_valid_splitByUser")
        uid_voc = os.path.join(data_path, dataset, "uid_voc.pkl")
        mid_voc = os.path.join(data_path, dataset, "mid_voc.pkl")
        cat_voc = os.path.join(data_path, dataset, "cat_voc.pkl")
        item_info = os.path.join(data_path, dataset, "item-info")
        reviews_info = os.path.join(data_path, dataset, "reviews-info")
    elif dataset == "content_wise":
        train_file = os.path.join(data_path, dataset, "local_train_splitByUser_cw")
        test_file = os.path.join(data_path, dataset, "local_test_splitByUser_cw")
        valid_file = os.path.join(data_path, dataset, "local_test_splitByUser_cw")
        uid_voc = os.path.join(data_path, dataset, "uid_voc_cw.pkl")
        mid_voc = os.path.join(data_path, dataset, "itemid_voc_cw.pkl")
        cat_voc = os.path.join(data_path, dataset, "series_voc_cw.pkl")
        item_info = os.path.join(data_path, dataset, "item-info_cw")
        reviews_info = os.path.join(data_path, dataset, "user_log_info")
    else:
        print("Please provide valid dataset!")

    model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        t1 = time.time()
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info,
                                  batch_size, maxlen, maxlen_hard, shuffle_each_epoch=False)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info,
                                 batch_size, maxlen, maxlen_hard)
        print('# Load data time (s):', round(time.time() - t1, 2))
        n_uid, n_mid, n_cat = train_data.get_n()

        model = model_selection(model_type, n_uid, n_mid, n_cat)
        model.restore(sess, model_path)
        print('test_auc: %.4f -- test_loss: %.4f -- test_accuracy: %.4f -- test_aux_loss: %.4f' % eval(
            sess, test_data, model, model_path, maxlen, maxlen_hard, model_type))


if __name__ == '__main__':
    if len(sys.argv) == 5:
        SEED = int(sys.argv[4])
    else:
        SEED = 1234
    # tf.set_random_seed(SEED)
    tf.compat.v1.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]
    if sys.argv[1] == 'train':
        train(model_type=sys.argv[2], seed=SEED)
    elif sys.argv[1] == 'test':
        test(model_type=sys.argv[2], seed=SEED)
    else:
        print('do nothing...')
