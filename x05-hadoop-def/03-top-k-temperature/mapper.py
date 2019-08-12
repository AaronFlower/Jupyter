#!/usr/bin/python

# -*- coding: utf-8 -*-

import sys

for line in sys.stdin:
    line = line.strip()
    if (not line or line.startswith('Wban')):
        continue

    _, day, temp, _ = line.split(',', 3)
    try:
        v = int(temp)
    except ValueError:
        continue
    print('%s\t%s' % (day, v))
