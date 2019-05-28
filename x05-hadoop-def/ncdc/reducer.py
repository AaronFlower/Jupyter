#!/usr/bin/env python

import sys

last_key, max_val = None, -9999
for line in sys.stdin:
    key, val = line.strip().split('\t')
    val = int(val)
    if key != last_key and last_key:
        print('\t'.join([last_key, str(max_val)]))
        last_key, max_val = key, val
    else:
        if max_val <= val:
            last_key = key
            max_val = val

if last_key:
    print('\t'.join([last_key, str(max_val)]))



