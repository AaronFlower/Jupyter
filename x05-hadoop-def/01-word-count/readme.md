## Hadoop Streaming

Hadoop Streaming 会自动解析 gzip 文件的，我们在使用 Hadoop Streaing 进可以不用关心解压缩的事情 。

### 测试

- mapper

```
# mapper.py

$ echo "foo foo quux labs foo bar quux"  | ./mapper.py

foo     1
foo     1
quux    1
labs    1
foo     1
bar     1
quux    1

```

- map and sort

```
$ echo "foo foo quux labs foo bar quux" |./mapper.py | sort
# or
$ echo "foo foo quux labs foo bar quux" |./mapper.py | sort -k 1
bar     1
foo     1
foo     1
foo     1
labs    1
quux    1
quux    1
```

- reducer

```
$ echo "foo foo quux labs foo bar quux" |./mapper.py | sort | ./reducer.py
bar     1
foo     3
labs    1
quux    2
```





#### 关于 `sort` 命令

```
❯ tldr sort

  sort

  Sort lines of text files.

  - Sort a file in ascending order:
    sort filename

  - Sort a file in descending order:
    sort -r filename

  - Sort a file in case-insensitive way:
    sort --ignore-case filename

  - Sort a file using numeric rather than alphabetic order:
    sort -n filename

  - Sort the passwd file by the 3rd field, numerically:
    sort -t: -k 3n /etc/passwd

  - Sort a file preserving only unique lines:
    sort -u filename

  - Sort human-readable numbers (in this case the 5th field of ls -lh):
    ls -lh | sort -h -k 5
```

### User Hadoop

```
❯ hdfs dfs -ls /user/eason/gutenberg
Found 3 items
-rw-r--r--   1 eason supergroup    1586393 2019-08-11 17:15 /user/eason/gutenberg/4300-0.txt
-rw-r--r--   1 eason supergroup    1428841 2019-08-11 17:15 /user/eason/gutenberg/5000-8.txt
-rw-r--r--   1 eason supergroup     674570 2019-08-11 17:15 /user/eason/gutenberg/pg20417.txt

```


```
$ hadoop jar /usr/local/Cellar/hadoop/3.1.2/libexec/share/hadoop/tools/lib/hadoop-streaming-3.1.2.jar \
-file ./mapper.py -mapper ./mapper.py \
-file ./reducer.py -reducer ./reducer.py \
-input /user/eason/gutenberg/*.txt -output /user/eason/gutenberg-output
```


如果使用 zsh 可能会报  `zsh on mathes` 错误，我们可以执行下 `setopt +o nomatch`  后再运行。

#### 查看运行结果

```
$ hdfs dfs -ls /user/eason/gutenberg-output
```
