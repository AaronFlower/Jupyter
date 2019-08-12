#!/usr/bin/python

# -*- coding: utf-8 -*-

import sys

max_date = None
max_temp = -999

for line in sys.stdin:
    line = line.strip()
    if (not line):
        continue

    date, temp = line.split('\t')

    try:
        temp = int(temp)
    except ValueError:
        continue

    if temp >= max_temp:
        max_temp = temp
        max_date = date

if max_date:
    print('%s\t%s' % (max_date, max_temp))
