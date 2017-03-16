'''
从一些网站的 API 获取数据
'''
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 字符串转日期
def strToDate(dateStr):
    return datetime.strptime(dateStr, '%Y%m%d')

# 获取股票数据
def getStockData(inc):
    url = 'http://chartapi.finance.yahoo.com/instrument/1.0/' + inc +'/chartdata;type=quote;range=10m/csv'
    data_source = urllib.request.urlopen(url).read().decode('utf-8')
    split_source = data_source.split('\n')
    stock_data = []
    for line in split_source:
        split_line = line.split(',')
        if (len(split_line) == 6) and 'close' not in line and 'labels' not in line:
            stock_data.append(line)
    date1, closep, highp, lowp, openp, volume = np.loadtxt(stock_data, delimiter = ',', unpack = True)
    dates = [strToDate(str(int(date))) for date in date1]
    return dates, closep, highp, lowp, openp, volume

# 普通 pyplot 来绘制。
def grap_data(inc, title = None):
    dates, closep, highp, lowp, openp, volume = getStockData(inc)
    plt.plot_date(dates, closep, '-', label='Close Price')
    plt.legend()
    plt.title(title or inc)
    plt.show()

# 用 sub_plot 来绘制。
def grap_data_figure(inc, title = None):
    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.plot_date(dates, closep,'-', label='Close Price')
    # x 轴上的 lable 调整。
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax1.grid(True)
    
    plt.legend()
    plt.title(title or inc)
    plt.subplots_adjust(left = 0.09,
                       bottom = 0.20,
                       right = 0.94,
                       top = 0.95,
                       wspace=0.2,
                       hspace=0.2)
    plt.show()

# 全用填充方式绘制出收益和损失
def grap_stock_balance(inc):
    dates, closep, highp, lowp, openp, volume = getStockData(inc)
    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.fill_between(dates, closep, closep[0], alpha=0.2, where=(closep > closep[0]), facecolor='g')
    ax1.fill_between(dates, closep, closep[0], alpha=0.2, where=(closep < closep[0]), facecolor='r')
    
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    
    # label trick
    ax1.plot([],[], linewidth = 5, label = 'Loss', color = 'r', alpha = 0.2)
    ax1.plot([],[], linewidth = 5, label = 'Gain', color = 'g', alpha = 0.2)
    plt.show()