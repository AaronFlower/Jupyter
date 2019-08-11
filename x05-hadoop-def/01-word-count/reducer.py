#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Reducer 从 stdin 中读取输入 <word> 1, 统计 word 出现的次数，然后输出到 stdout
"""

import sys

current_word = None
current_count = 0
word = None

for line in sys.stdin:
    line = line.strip()
    word, count = line.split('\t', 1)

    try:
        count = int(count)
    except ValueError:
        continue

    # Hadoop 会将 mapper.py 的输出进行 sort ，所以 reducer.py 从 stdin 读取
    # 是有顺序的，即单词按序列读入的，即是 word 是 key.
    if current_word == word:
        current_count += count
    else:
        if current_word:
            print('%s\t%s' % (current_word, current_count))
        current_word = word
        current_count = count

# 别忘记输出最后一个 word
if current_word == word:
    print('%s\t%s' % (current_word, current_count))
