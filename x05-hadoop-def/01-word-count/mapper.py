#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
从 STDIN 中一行行读取数据，直接输出 <word> 1 到 STDOUT 供 reducer 使用
"""

import sys

for line in sys.stdin:
    # remove leading and trailing whitespace , then split the line into words.
    words = line.strip().split()
    for word in words:
        print('%s\t%s' % (word, 1))
