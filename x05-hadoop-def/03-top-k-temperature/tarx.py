#!/usr/local/bin/python3

import os
import tarfile

for filename in os.listdir('.'):
    if filename.endswith('tar.gz'):
        month, _ = filename.split('.', 1)
        daily = '%sdaily.txt' % month
        try:
            tf = tarfile.open(filename)
            tf.extract(daily, './2000/')
        except Exception:
            continue
