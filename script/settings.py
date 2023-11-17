# -*- coding: utf-8 -*-
EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
POS_EMBEDDING_DIM = 2

# train.py
# model_type='TGIN'
data_path='dataset'
dataset='content_wise'
tri_data='wnd3_alpha_01_theta_09_tri_num_10' 
n_tri=10
batch_size=256
maxlen=30   # 20
maxlen_hard=20
test_iter=100
save_iter=100
lr=1e-3

# model.py
n_neg = 5
use_dice = True
use_negsampling = True

# tgin.py
single_tri_agg_flag = 'reduce_mean'
# multi_tri_agg_flag = 'weighted_sum'
multi_tri_agg_flag = 'mhsa'
