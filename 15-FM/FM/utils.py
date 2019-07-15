# -*- coding: utf-8 -*-

import pandas as pd
import pickle
import logging

# 使用坐标来存储稀疏矩阵，Coordinate list(COO)
from scipy.sparse import coo_matrix

# 使用内置的 logging , 设置 logging 格式
FORMAT = '%(asctime)s : %(levelname)s : %(messages)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
