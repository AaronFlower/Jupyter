#!/usr/bin/env python

import sys

for line in sys.stdin:
    line = line.strip()
    # year, temp, q = val[15,4], val[87,5], val[92,1]
    year, temp, q = line[15:19], line[87:93], line[92:93]
    if temp != '+9999' and (q in '01459'):
        print('\t'.join([year, temp]))
