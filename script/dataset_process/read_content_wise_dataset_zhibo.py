#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/2 15:14

import numpy as np
import scipy as sp
import dask.dataframe as ddf

file_name = ".../ContentWiseImpressions/data/CW10M/splits/urm_items.train.npz"
data = np.load(file_name)

print(data)

data2 = sp.sparse.load_npz(file_name)
print(data2.toarray())

# read parquet data files
file_name_interaction = ".../ContentWiseImpressions/data/CW10M/interactions"
file_name_imp_direct_link = ".../ContentWiseImpressions/data/CW10M/impressions-direct-link"
file_name_imp_non_direct_link = ".../ContentWiseImpressions/data/CW10M/impressions-non-direct-link"

data_interaction = ddf.read_parquet(file_name_interaction, engine="pyarrow")
data_imp_direct_link = ddf.read_parquet(file_name_imp_direct_link, engine="pyarrow")
data_imp_non_direct_link = ddf.read_parquet(file_name_imp_non_direct_link, engine="pyarrow")

print(data_interaction)
print("data_interaction shape: {}".format(data_interaction.shape))
print(data_interaction.head())
# print(data_imp_direct_link)
# print(data_imp_non_direct_link)