#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/3 22:38

import  os

data_dir = "processed_data"

fin = open(os.path.join(data_dir, "jointed-new-split-info-cw"), "r")
ftrain = open(os.path.join(data_dir, "local_train_cw"), "w")
ftest = open(os.path.join(data_dir, "local_test_cw"), "w")

last_user = "0"
common_fea = ""
line_idx = 0
item_id_list = []
series_id_list = []
time_stamp_list = []
for line in fin:
    elements = line.strip().split('\t')
    ds = elements[0]
    clk = int(elements[1])
    user = elements[2]
    item_id = elements[3]
    dt = elements[-1]
    series_id = elements[4]
    time_stamp = elements[12]

    if ds == "20180118":
        fo = ftrain
    else:
        fo = ftest

    if user != last_user:
        item_id_list = []
        series_id_list = []
        time_stamp_list =[]
    else:
        history_clk_num = len(item_id_list)
        item_str = ""
        series_str = ""
        time_stamp_str = ""
        for s in series_id_list:
            series_str += s + ""
        for i in item_id_list:
            item_str += i + ""
        for t in time_stamp_list:
            time_stamp_str += t + ""
        if len(series_str) > 0: series_str = series_str[:-1]  # remove last char ""
        if len(item_str) > 0: item_str = item_str[:-1]
        if len(time_stamp_str) >0 :time_stamp_str = time_stamp_str[:-1]
        if history_clk_num >= 10:
        # if history_clk_num >= 1:
            # 把用户历史行为的item和series也聚合在了一起。
            print(str(clk) + "\t" + user + "\t" + item_id + "\t" + series_id + "\t" + time_stamp + "\t" + item_str + "\t" + series_str + "\t" + time_stamp_str, end="\n", file=fo)
    last_user = user
    if clk:
        item_id_list.append(item_id)
        series_id_list.append(series_id)
        time_stamp_list.append(time_stamp)

    line_idx += 1