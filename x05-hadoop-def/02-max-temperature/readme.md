## Max Temperature

```
$ cat sample.txt| ./mapper.py
20001201        45
20001202        34
20001203        45
20001204        50
20001205        51
20001206        47
20001207        53
20001208        48
20001209        57

$ cat sample.txt| ./mapper.py | ./reducer.py
20001209        57

```

### 上传到 Hadoop 上计算

1. 上传数据文件

```
hdfs dsf -mkdir /user/eason/ncdc
hdfs dsf -copyFromLocal ./2012*.txt /user/eason/ncdc/
```

2. 执行计算

```
$ hadoop jar /usr/local/Cellar/hadoop/3.1.2/libexec/share/hadoop/tools/lib/hadoop-streaming-3.1.2.jar \
-file ./mapper.py -mapper ./mapper.py \
-file ./reducer.py -reducer ./reducer.py \
-input /user/eason/ncdc/*.txt -output /user/eason/ncdc-output
```
