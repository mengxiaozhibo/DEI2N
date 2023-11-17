#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/3 22:40

import pickle
import os

data_dir = "processed_data"

f_train = open(os.path.join(data_dir, "local_train_splitByUser_cw"), "r")
uid_dict = {}
itemid_dict = {}
series_dict = {}

iddd = 0
for line in f_train:
    elements = line.strip("\n").split("\t")
    clk = elements[0]
    uid = elements[1]
    iid = elements[2]
    series_id = elements[3]
    item_id_list = elements[4]
    series_id_list = elements[5]
    if uid not in uid_dict:
        uid_dict[uid] = 0
    uid_dict[uid] += 1
    if iid not in itemid_dict:
        itemid_dict[iid] = 0
    itemid_dict[iid] += 1
    if series_id not in series_dict:
        series_dict[series_id] = 0
    series_dict[series_id] += 1

    for i in item_id_list.split(""):
        if i not in itemid_dict:
            itemid_dict[i] = 0
        itemid_dict[i] += 1
    iddd += 1
    for s in series_id_list.split(""):
        if s not in series_dict:
            series_dict[s] = 0
        series_dict[s] += 1

# 经常出现的排在前面
sorted_uid_dict = sorted(uid_dict.items(), key=lambda x: x[1], reverse=True)
sorted_itemid_dict = sorted(itemid_dict.items(), key=lambda x: x[1], reverse=True)
sorted_series_dict = sorted(series_dict.items(), key=lambda x: x[1], reverse=True)

uid_voc = {}
index = 0
for key, value in sorted_uid_dict:
    uid_voc[key] = index
    index += 1

itemid_voc = {}
itemid_voc['default_item_id'] = 0
index = 1
for key, value in sorted_itemid_dict:
    itemid_voc[key] = index
    index += 1

series_voc = {}
series_voc['default_series'] = 0
index = 1
for key, value in sorted_series_dict:
    series_voc[key] = index
    index += 1

# for python3
pickle.dump(uid_voc, open(os.path.join(data_dir, "uid_voc_cw.pkl"), "wb"))
pickle.dump(itemid_voc, open(os.path.join(data_dir, "itemid_voc_cw.pkl"), "wb"))
pickle.dump(series_voc, open(os.path.join(data_dir, "series_voc_cw.pkl"), "wb"))