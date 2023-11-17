#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/3 22:39

import random
import os

data_dir = "processed_data"

fi = open(os.path.join(data_dir, "local_train_cw"), "r")
ftrain = open(os.path.join(data_dir, "local_train_splitByUser_cw"), "w")
fvalid = open(os.path.join(data_dir, "local_valid_splitByUser_cw"), "w")
ftest = open(os.path.join(data_dir, "local_test_splitByUser_cw"), "w")
while True:
    rand_int = random.randint(1, 10)
    noclk_line = fi.readline().strip()
    clk_line = fi.readline().strip()
    if noclk_line == "" or clk_line == "":
        break
    if noclk_line != "" or clk_line != "":
        clk_line_str = clk_line.strip("\n").split("\t")
        time_stamp_target = clk_line_str[4]
        time_stamp_trigger = clk_line_str[7].split("")[-1]
        if int(time_stamp_target) - int(time_stamp_trigger)>28800000:
            continue
    if rand_int == 2:
        print(noclk_line, end="\n", file=ftest)  # for python3
        print(clk_line, end="\n", file=ftest)  # for python3
    elif rand_int == 4:
        print(noclk_line, end="\n", file=fvalid)  # for python3
        print(clk_line, end="\n", file=fvalid)  # for python3
    else:
        print(noclk_line, end="\n", file=ftrain)  # for python3
        print(clk_line, end="\n", file=ftrain)  # for python3