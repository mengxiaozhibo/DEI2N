#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/3 22:33

import random
import os
import dask.dataframe as ddf

data_dir = "processed_data"


def read_cw_datasets():
    # parquet data files
    file_name_interaction = ".../ContentWiseImpressions/data/CW10M/interactions"
    file_name_imp_direct_link = ".../ContentWiseImpressions/data/CW10M/impressions-direct-link"
    file_name_imp_non_direct_link = ".../ContentWiseImpressions/data/CW10M/impressions-non-direct-link"

    data_interaction = ddf.read_parquet(file_name_interaction, engine="pyarrow")
    # data_imp_direct_link = ddf.read_parquet(file_name_imp_direct_link, engine="pyarrow")
    # data_imp_non_direct_link = ddf.read_parquet(file_name_imp_non_direct_link, engine="pyarrow")

    # sample 10% records in the original data
    data_interaction_sampled = data_interaction.sample(frac=0.25, random_state=132)

    print(data_interaction_sampled)
    print("data_interaction shape: {}".format(data_interaction_sampled.shape))
    print(data_interaction_sampled.head())

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # format: [unixtime, user_id, item_id, series_id, episode_number, series_length, item_type, recommendation_id,
    # interaction_type, version_factor, explicit_rating]
    data_interaction_sampled.to_csv(os.path.join(data_dir, 'data_interaction_sampled.csv'), single_file=True)

    # reformat the data, put the unixtime at the end
    # format: [user_id, item_id, series_id, episode_number, series_length, item_type, recommendation_id,
    #          interaction_type, version_factor, explicit_rating, unixtime]
    fi = open(os.path.join(data_dir, 'data_interaction_sampled.csv'), "r")
    fo = open(os.path.join(data_dir, "user_log_info"), "w")
    for line in fi:
        elements = line.strip().split(",")
        if elements[0] == "utc_ts_milliseconds":
            continue

        elements_new = elements[1:]
        elements_new.append(elements[0])
        # format as [user, item, ... , timestamp]
        print("\t".join(elements_new), end="\n", file=fo)


def process_meta():
    fi = open(os.path.join(data_dir, "user_log_info"), "r")
    fo = open(os.path.join(data_dir, "item-info_cw"), "w")
    item_info = dict()
    for line in fi:
        elements = line.strip().split("\t")
        item_id = elements[1]
        series_id = elements[2]
        item_info[item_id] = series_id

    for item_id, series_id in item_info.items():
        print(item_id + "\t" + series_id, end="\n", file=fo)


def manual_join():
    # aggregate user behaviors
    user_map = {}
    item_list = []
    item_info_map = {}
    with open(os.path.join(data_dir, 'user_log_info'), "r") as file:
        for line in file:
            # format: [user_id, item_id, series_id, episode_number, series_length, item_type, recommendation_id,
            #          interaction_type, version_factor, explicit_rating, unixtime]
            elements = line.strip().split("\t")

            # aggregate user behaviors
            if elements[0] not in user_map:
                user_map[elements[0]] = []
            user_map[elements[0]].append(("\t".join(elements), float(elements[-1])))

            item_list.append(elements[1])

            # store item information [series_id, episode_number, series_length, item_type]
            if elements[1] not in item_info_map:
                item_info_map[elements[1]] = elements[2:6]

    # sampling the negative samples
    fo = open(os.path.join(data_dir, "jointed-new-cw"), "w")
    for key in user_map:
        # sort the user behavior items by unix timestamp
        sorted_user_hb = sorted(user_map[key], key=lambda x: x[1])
        for line, t in sorted_user_hb:
            elements = line.split("\t")
            item = elements[1]
            j = 0
            while True:
                # sampling the negative samples
                item_neg_index = random.randint(0, len(item_list) - 1)
                item_neg = item_list[item_neg_index]
                if item_neg == item:
                    continue
                elements[1] = item_neg
                elements[2:6] = item_info_map[item_neg]  # store the negative item information
                print("0" + "\t" + "\t".join(elements), end="\n", file=fo)
                j += 1
                if j == 1:  # negative sampling frequency
                    break
            print("1" + "\t" + line, end="\n", file=fo)


# put the split mark
def split_test():
    fi = open(os.path.join(data_dir, "jointed-new-cw"), "r")
    fo = open(os.path.join(data_dir, "jointed-new-split-info-cw"), "w")
    # 统计user在行为表里出现的次数
    user_count = {}
    for line in fi:
        elements = line.strip().split('\t')
        user = elements[1]
        if user not in user_count:
            user_count[user] = 0
        user_count[user] += 1
    fi.seek(0)

    i = 0
    last_user = "abcd1234"
    for line in fi:
        line = line.strip()
        elements = line.split("\t")
        user = elements[1]
        if user == last_user:
            if i < user_count[user] - 2:  # 1 +negative samples
                print("20180118" + "\t" + line, end="\n", file=fo)  # for python3
            else:  # 每个用户的最后一个行为和其负样本
                print("20190119" + "\t" + line, end="\n", file=fo)  # for python3
        else:
            last_user = user
            i = 0
            if i < user_count[user] - 2:
                print("20180118" + "\t" + line, end="\n", file=fo)  # for python3
            else:  # 每个用户的最后一个行为和其负样本
                print("20190119" + "\t" + line, end="\n", file=fo)  # for python3
        i += 1


read_cw_datasets()
process_meta()
manual_join()
split_test()