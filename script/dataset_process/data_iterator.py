# -*- coding: utf-8 -*-
# Copyright (C) 2016-2018 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import numpy
import json
#import cPickle as pkl
import _pickle as pkl
import random
import gzip
from dataset_process import shuffle


def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key, value) in d.items())


def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)
            #return unicode_to_utf8(pkl.load(f))


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class DataIterator:
    def __init__(self, source,
                 uid_voc,
                 mid_voc,
                 cat_voc,
                 item_info,
                 reviews_info,
                 batch_size=128,
                 maxlen=20,
                 maxlen_hard=10,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 max_batch_size=20,
                 minlen=None):
        # shuffle the input file
        if shuffle_each_epoch:
            self.source_orig = source
            self.source = shuffle.main(self.source_orig, temporary=True)
        else:
            self.source = fopen(source, 'r')
        self.source_dicts = []
        for source_dict in [uid_voc, mid_voc, cat_voc]:
            self.source_dicts.append(load_dict(source_dict))

        # Mapping Dict: {item:category}
        f_meta = open(item_info, "r")
        meta_map = {}
        for line in f_meta:
            arr = line.strip().split("\t")
            if arr[0] not in meta_map:
                meta_map[arr[0]] = arr[1]
        self.meta_id_map = {}
        for key in meta_map:
            val = meta_map[key]
            if key in self.source_dicts[1]:
                mid_idx = self.source_dicts[1][key]
            else:
                mid_idx = 0
            if val in self.source_dicts[2]:
                cat_idx = self.source_dicts[2][val]
            else:
                cat_idx = 0
            self.meta_id_map[mid_idx] = cat_idx

        # Get all the interacted items
        f_review = open(reviews_info, "r") #[user, item, rating, timestamp]
        self.mid_list_for_random = []
        for line in f_review:
            arr = line.strip().split("\t")
            tmp_idx = 0
            if arr[1] in self.source_dicts[1]: # if the item exsist,
                tmp_idx = self.source_dicts[1][arr[1]] # get item's ID
            self.mid_list_for_random.append(tmp_idx) # list of all the interacted items

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.maxlen_hard = maxlen_hard
        self.minlen = minlen
        self.skip_empty = skip_empty
        self.n_uid = len(self.source_dicts[0])
        self.n_mid = len(self.source_dicts[1])
        self.n_cat = len(self.source_dicts[2])

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * max_batch_size

        self.end_of_data = False

    def get_n(self):
        return self.n_uid, self.n_mid, self.n_cat

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.source = shuffle.main(self.source_orig, temporary=True)
        else:
            self.source.seek(0)

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        target_aux = []

        # Buffer: ss is one line of local_train_splitByUser/local_test_splitByUser
        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))

            # sort by history behavior length
            if self.sort_by_length:
                his_length = numpy.array([len(s[5].split("")) for s in self.source_buffer])
                tidx = his_length.argsort()
                _sbuf = [self.source_buffer[i] for i in tidx]
                self.source_buffer = _sbuf
            else:
                self.source_buffer.reverse()

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            '''
            each ss, [label, user, item, category, time_stamp, [item list], [item cate list], [time_stamp_list]
            [['0',
              'AZPJ9LUT0FEPY',
              'B00AMNNTIA',
              'Literature & Fiction',
              '0307744434\x020062248391\x020470530707\x020978924622\x021590516400',
              'Books\x02Books\x02Books\x02Books\x02Books']]
            '''
            while True:
                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                uid = self.source_dicts[0][ss[1]] if ss[1] in self.source_dicts[0] else 0
                mid = self.source_dicts[1][ss[2]] if ss[2] in self.source_dicts[1] else 0
                cat = self.source_dicts[2][ss[3]] if ss[3] in self.source_dicts[2] else 0
                time_stamp = int(int(ss[4])/1000)
                tmp = []
                for fea in ss[5].split(""):
                    m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
                    tmp.append(m)
                mid_list = tmp

                tmp1 = []
                for fea in ss[6].split(""):
                    c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                    tmp1.append(c)
                cat_list = tmp1

                # get the time_stamp
                tmp2 = []
                for fea in ss[7].split(""):
                    c = int(time_stamp/60 - int(fea)/1000/60)
                    if c>=86400:
                        c=86400
                    tmp2.append(c)
                time_stamp_list = tmp2

                #get the trigger item
                trigger_mid = []
                trigger_cate= []

                if time_stamp_list[-1] <= 480:
                    trigger_mid.append(mid_list[-1])
                    trigger_cate.append(cat_list[-1])
                else:
                    continue

                #generate the hard seq
                tmp_mid = []
                tmp_cat = []
                tmp_time =[]

                for index,cat_value in enumerate(cat_list):
                    if cat_value == trigger_cate[0]:
                        tmp_mid.append(mid_list[index])
                        tmp_cat.append(cat_value)
                        tmp_time.append(time_stamp_list[index])
                mid_hard_list = tmp_mid
                cat_hard_list = tmp_cat
                time_stamp_hard_list = tmp_time
                # print(len(mid_hard_list))

                # read from source file and map to word index
#                if len(mid_list) > self.maxlen:
 #                  mid_list = mid_list[-self.maxlen:]
 #                  cat_list = cat_list[-self.maxlen:]
 #               elif len(mid_list) < self.maxlen:
 #                   bu_len=self.maxlen-len(mid_list)
 #                   mid_list+=[0]*bu_len
 #                   cat_list+=[0]*bu_len

#                if len(mid_list) > self.maxlen:
 #                   continue
  #              while len(mid_list) < self.maxlen:
   #                 mid_list.append(0)
   #                 cat_list.append(0)
                if self.minlen != None:
                    if len(mid_list) <= self.minlen:
                        continue
                if self.skip_empty and (not mid_list):
                    continue

                #-------------------------------- Negative sample -------------------------------#
                noclk_mid_list = []
                noclk_cat_list = []
                for pos_mid in mid_list:
                    noclk_tmp_mid = []
                    noclk_tmp_cat = []
                    noclk_index = 0
                    while True:
                        # Random sample negative item for (history records) mid_list
                        # Including item+category，luwei: mid_list_for_random是按照reviews-info里的分布来得到的。所以负采样的时候也是按照这个分布
                        noclk_mid_indx = random.randint(0, len(self.mid_list_for_random) - 1)
                        noclk_mid = self.mid_list_for_random[noclk_mid_indx]
                        if noclk_mid == pos_mid:
                            continue
                        noclk_tmp_mid.append(noclk_mid)
                        noclk_tmp_cat.append(self.meta_id_map[noclk_mid])
                        noclk_index += 1
                        if noclk_index >= 5:
                            break
                    noclk_mid_list.append(noclk_tmp_mid)
                    noclk_cat_list.append(noclk_tmp_cat)
                #print("mid_list",len(mid_list))
                #print("cat_list",len(cat_list))
                #print("noclk_mid_list",len(noclk_mid_list))
               # print("noclk_cat_list",len(noclk_cat_list))
                source.append([uid, mid, cat, time_stamp, trigger_mid, trigger_cate,  mid_list, cat_list, time_stamp_list, mid_hard_list, cat_hard_list, time_stamp_hard_list, noclk_mid_list, noclk_cat_list])
                target.append([float(ss[0]), 1-float(ss[0])])
                if cat==trigger_cate[0]:
                    target_aux.append([1.0, 0.0])
                else:
                    target_aux.append([0.0, 1.0])

                if len(source) >= self.batch_size or len(target) >= self.batch_size or len(target_aux) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target, target_aux = self.__next__()

        return source, target, target_aux