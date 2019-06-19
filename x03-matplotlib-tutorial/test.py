#coding:utf8
'''
从一些网站的 API 获取数据
'''
import urllib.request

def grap_data(inc):
    url = 'http://chartapi.finance.yahoo.com/instrument/1.0/' + inc + '/chartdata;type=quote;range=1d/csv'
    data_source = urllib.request.urlopen(url).read().decode()
    split_source = data_source.split('\n')
    stock_data = []
    print(data_source)
    for line in split_source:
        split_line = line.split(',')
        if (len(split_line) == 6) and 'close' not in split_line:
            stock_data.append(split_line)
    print(stock_data)
            
grap_data('TSLA')