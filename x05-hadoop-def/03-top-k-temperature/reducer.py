#!/usr/local/bin/python3

# -*- coding: utf-8 -*-

import sys
import queue


class DayTemp(object):

    def __init__(self, date, temp):
        self.date = date
        self.temp = temp

    def __lt__(self, other):
        return self.temp < other.temp


k = 3
q = queue.PriorityQueue()

for line in sys.stdin:
    line = line.strip()
    if (not line):
        continue

    date, temp = line.split('\t')

    try:
        temp = int(temp)
    except ValueError:
        continue

    q.put(DayTemp(date, temp))

    if q.qsize() > k:
        q.get()

while not q.empty():
    one_day = q.get()
    print('%s\t%s' % (one_day.date, one_day.temp))
