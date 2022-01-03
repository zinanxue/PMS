holt +

import numpy as np

from datetime import datetime
import time
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import warnings
warnings.simplefilter('ignore')
from sklearn.metrics import mean_squared_error

from impala.dbapi import connect
from math import sqrt

from ehsy_ds_config import hive_host, hive_port, hive_user, hive_password

def RMSE(params, *args):
    Y = args[0]
    type = args[1]
    rmse = 0
    alpha, beta, gamma = params
    m = args[2]
    a = [sum(Y[0:m]) / float(m)]
    b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
    if type == 'additive':
        s = [Y[i] - a[0] for i in range(m)]
        y = [a[0] + b[0] + s[0]]
    for i in range(len(Y)):
        a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
        y.append(a[i + 1] + b[i + 1] + s[i + 1])
        rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y, y[:-1])]) / len(Y))
    return rmse


def MSE(a, b):
    return np.mean(np.power(a - b, 2))


if __name__ == '__main__':

    # 20211202

    exec_pt = '20211230'

    pyads = connect(hive_host, hive_port, 'ehsy', auth_mechanism='PLAIN', user=hive_user, password=hive_password)

    cur = pyads.cursor()
    # sql = '''
    #     select sku_code,60_ago,59_ago,58_ago,57_ago,56_ago,55_ago,54_ago,53_ago,52_ago,51_ago,50_ago
    #             ,49_ago,48_ago,47_ago,46_ago,45_ago,44_ago,43_ago,42_ago,41_ago,40_ago
    #             ,39_ago,38_ago,37_ago,36_ago,35_ago,34_ago,33_ago,32_ago,31_ago,30_ago
    #             ,29_ago,28_ago,27_ago,26_ago,25_ago,24_ago,23_ago,22_ago,21_ago,20_ago
    #             ,19_ago,18_ago,17_ago,16_ago,15_ago,14_ago,13_ago,12_ago,11_ago,10_ago
    #             ,9_ago,8_ago,7_ago,6_ago,5_ago,4_ago,3_ago,2_ago,1_ago
    #             from ehsy.ads_bh_sale_report_stats_country_all_t
    # '''


    # demo数据
    sql = '''
        select sku_code, 12_ago_quantity, 11_ago_quantity, 10_ago_quantity,
            9_ago_quantity, 8_ago_quantity, 7_ago_quantity, 6_ago_quantity, 
            5_ago_quantity, 4_ago_quantity, 3_ago_quantity, 2_ago_quantity,
            1_ago_quantity
        from ehsy.ods_bh_sale_report_stats_country_lv1
    '''

    cur.execute(sql)
    cur_list = cur.fetchall()
    
    
    insert_sql = '''
        insert into table ehsy.ods_pms_sku_forecast_hotwinter_plus partition (pt='{0}') 
        values
        {1}
    '''
    
    relt_rows = []

    idx = 1
    for row in cur_list:
        print('-=-=-=-=-=-=-= begin {0} -- {1}'.format(row[0], idx))
        # 前53个月
        data = row[1:(len(row) - 3)]
        # 最后7个月的
        data_rdd = row[(len(row) - 3):]

        fit1 = ExponentialSmoothing(data, seasonal_periods=6, trend='add', seasonal='add').fit()
        # 预测7个月的
        forecast_fit1 = np.maximum(fit1.forecast(3), 0)
        # 向上取整
        forecast_fit1_ceil = np.ceil(forecast_fit1)
        # 向下取整
        forecast_fit1_floor = np.floor(forecast_fit1)

        rms1_ceil = sqrt(mean_squared_error(data_rdd, list(forecast_fit1_ceil)))
        rms1_floor = sqrt(mean_squared_error(data_rdd, list(forecast_fit1_floor)))

        # 趋势成分 抑制
        fit2 = ExponentialSmoothing(data, seasonal_periods=6, trend='add', seasonal='add', damped_trend=True).fit()
        forecast_fit2 = np.maximum(fit1.forecast(3), 0)
        # 向上取整的
        forecast_fit2_ceil = np.ceil(forecast_fit2)
        # 向下取整
        forecast_fit2_floor = np.floor(forecast_fit2)

        rms2_ceil = sqrt(mean_squared_error(data_rdd, list(forecast_fit2_ceil)))
        rms2_floor = sqrt(mean_squared_error(data_rdd, list(forecast_fit2_floor)))

        rms_k = [rms1_ceil, rms1_floor, rms2_ceil, rms2_floor]
        rms_v = [forecast_fit1_ceil, forecast_fit1_floor, forecast_fit2_ceil, forecast_fit2_floor]
        rms_index = np.argmin(rms_k)

        # relt_rows.append("('{0}', {1}, {2}, {3})".format(row[0], ','.join([str(_) for _ in list(row)[-12:]]), ','.join([str(_) for _ in rms_v[rms_index].tolist()]), datetime.strftime(datetime.now(), '%Y%m%d')))

        relt_rows.append("('{0}', {1}, {2}, 0, 0, 0, 0, {3}, {4}, null)".format(row[0], ','.join([str(_) for _ in list(row)[-12:]]), ','.join([str(_) for _ in rms_v[rms_index].tolist()]), exec_pt, rms_k[rms_index]))

        print('-----end {0} -----'.format(row[0]))
        idx += 1

    cur.execute('set hive.exec.dynamic.partition=true;')
    cur.execute('set hive.exec.dynamic.partition.mode=nonstrict;')

    cur.execute(insert_sql.format(exec_pt, ','.join(relt_rows)))

    # cur.execute(insert_sql.format(time.strftime('%Y%m%d', time.localtime(time.time())), ','.join(relt_rows)))
    cur.close()
    print('-------------完成----------------')



-------------------------------------------------------------------------holt x-------------------------------



import numpy as np

from ehsy_ds_config import hive_host, hive_port, hive_user, hive_password
from datetime import datetime
import time

from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import warnings
warnings.simplefilter('ignore')
from sklearn.metrics import mean_squared_error

from impala.dbapi import connect
from math import sqrt



def RMSE(params, *args):
    Y = args[0]
    type = args[1]
    rmse = 0
    alpha, beta, gamma = params
    m = args[2]
    a = [sum(Y[0:m]) / float(m)]
    b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
    if type == 'additive':
        s = [Y[i] - a[0] for i in range(m)]
        y = [a[0] + b[0] + s[0]]
    for i in range(len(Y)):
        a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
        y.append(a[i + 1] + b[i + 1] + s[i + 1])
        rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y, y[:-1])]) / len(Y))
    return rmse


def MSE(a, b):
    return np.mean(np.power(a - b, 2))


if __name__ == '__main__':

    # 20211202
    exec_pt = '20211229'

    pyads = connect(hive_host, hive_port, 'ehsy', auth_mechanism='PLAIN', user=hive_user, password=hive_password)
    cur = pyads.cursor()

    # sql = '''
    #     select sku_code,60_ago,59_ago,58_ago,57_ago,56_ago,55_ago,54_ago,53_ago,52_ago,51_ago,50_ago
    #             ,49_ago,48_ago,47_ago,46_ago,45_ago,44_ago,43_ago,42_ago,41_ago,40_ago
    #             ,39_ago,38_ago,37_ago,36_ago,35_ago,34_ago,33_ago,32_ago,31_ago,30_ago
    #             ,29_ago,28_ago,27_ago,26_ago,25_ago,24_ago,23_ago,22_ago,21_ago,20_ago
    #             ,19_ago,18_ago,17_ago,16_ago,15_ago,14_ago,13_ago,12_ago,11_ago,10_ago
    #             ,9_ago,8_ago,7_ago,6_ago,5_ago,4_ago,3_ago,2_ago,1_ago
    #             from ehsy.ads_bh_sale_report_stats_country_all_t
    # '''

    # demo数据
    sql = '''
        select sku_code, 12_ago_quantity, 11_ago_quantity, 10_ago_quantity,
            9_ago_quantity, 8_ago_quantity, 7_ago_quantity, 6_ago_quantity, 
            5_ago_quantity, 4_ago_quantity, 3_ago_quantity, 2_ago_quantity,
            1_ago_quantity
        from ehsy.ods_bh_sale_report_stats_country_lv1
    '''

    cur.execute(sql)
    cur_list = cur.fetchall()

    insert_sql = '''
        insert into table ehsy.ods_pms_sku_forecast_hotwinter_multiplication partition (pt='{0}') 
        values
        {1}
    '''
    
    relt_rows = []
    idx = 1
    for row in cur_list:
        print('-=-=-=-=-=-=-= begin {0} -- {1}'.format(row[0], idx))
        # 前53个月
        data = row[1:(len(row) - 3)]
        # 最后7个月的
        data_rdd = row[(len(row) - 3):]

        try:
            fit1 = ExponentialSmoothing(data, seasonal_periods=4, trend='mul', seasonal='mul').fit()
        except Exception as e:
            print('----{0} exception: {1}------'.format(row[0], e.args[0]))
        else:
            # 预测7个月的
            forecast_fit1 = np.maximum(fit1.forecast(3), 0)
            # 向上取整
            forecast_fit1_ceil = np.ceil(forecast_fit1)
            # 向下取整
            forecast_fit1_floor = np.floor(forecast_fit1)

            rms1_ceil = sqrt(mean_squared_error(data_rdd, list(forecast_fit1_ceil)))
            rms1_floor = sqrt(mean_squared_error(data_rdd, list(forecast_fit1_floor)))

            # 趋势成分 抑制
            fit2 = ExponentialSmoothing(data, seasonal_periods=4, trend='mul', seasonal='mul', damped_trend=True).fit()
            forecast_fit2 = np.maximum(fit1.forecast(3), 0)
            # 向上取整的
            forecast_fit2_ceil = np.ceil(forecast_fit2)
            # 向下取整
            forecast_fit2_floor = np.floor(forecast_fit2)

            rms2_ceil = sqrt(mean_squared_error(data_rdd, list(forecast_fit2_ceil)))
            rms2_floor = sqrt(mean_squared_error(data_rdd, list(forecast_fit2_floor)))

            rms_k = [rms1_ceil, rms1_floor, rms2_ceil, rms2_floor]
            rms_v = [forecast_fit1_ceil, forecast_fit1_floor, forecast_fit2_ceil, forecast_fit2_floor]
            rms_index = np.argmin(rms_k)

            # relt_rows.append("('{0}', {1}, {2}, {3})".format(row[0], ','.join([str(_) for _ in list(row)[-12:]]), ','.join([str(_) for _ in rms_v[rms_index].tolist()]), datetime.strftime(datetime.now(), '%Y%m%d')))

            relt_rows.append("('{0}', {1}, {2}, 0, 0, 0, 0, {3}, {4}, null)".format(row[0], ','.join([str(_) for _ in list(row)[-12:]]), ','.join([str(_) for _ in rms_v[rms_index].tolist()]), exec_pt, rms_k[rms_index]))
        finally:
            print('-----end {0} -----'.format(row[0]))
            idx += 1

    if len(relt_rows) > 0:
        cur.execute('set hive.exec.dynamic.partition=true;')
        cur.execute('set hive.exec.dynamic.partition.mode=nonstrict;')
        cur.execute(insert_sql.format(exec_pt, ','.join(relt_rows)))
    cur.close()
    print('-------------完成----------------')


-----------------------------arima1---------

# arima处理数据到Hive

import datetime
import time

from sklearn import metrics
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
from impala.dbapi import connect
import matplotlib.pyplot as plt
import csv

pyads = connect('hiveserver2.ehsy.com', 10000, 'ehsy', auth_mechanism='PLAIN', user='readonly', password='Ehsyol@2021')

#
#

def getMa():
    f = open('MA.csv', 'w', encoding='utf-8', newline='')
    yy = open('M.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_wri = csv.writer(yy)
    date_col = ['2020-12-01', '2021-01-01', '2021-02-01',
                '2021-03-01', '2021-04-01', '2021-05-01',
                '2021-06-01', '2021-07-01', '2021-08-01',
                '2021-09-01', '2021-10-01', '2021-11-01']
    # 列名
    col = ['sku_code','2020-12-01', '2021-01-01', '2021-02-01',
           '2021-03-01', '2021-04-01', '2021-05-01',
           '2021-06-01', '2021-07-01', '2021-08-01',
           '2021-09-01', '2021-10-01', '2021-11-01']
    col2 = ['sku_code', '2020-12-01', '2021-01-01', '2021-02-01',
            '2021-03-01', '2021-04-01', '2021-05-01',
            '2021-06-01', '2021-07-01', '2021-08-01',
            '2021-09-01', '2021-10-01', '2021-11-01']
    py = pyads.cursor()

    sql = "select sku_code,12_ago_quantity,11_ago_quantity,10_ago_quantity,9_ago_quantity" \
          ",8_ago_quantity ,7_ago_quantity,6_ago_quantity,5_ago_quantity,4_ago_quantity" \
          ",3_ago_quantity,2_ago_quantity ,1_ago_quantity " \
          "from ehsy.ods_bh_sale_report_stats_country_lv1  where model='ARIMA'"
    print(sql)
    py.execute(sql)
    # 将py_hive查询的数据给到pandas
    df = pd.DataFrame(py.fetchall(), columns=col)
    # 重命名列名
    df.rename(columns={
        '12_ago_quantity': '2020-12-01',
        '11_ago_quantity': '2021-01-01',
        '10_ago_quantity': '2021-02-01',
        '9_ago_quantity': '2021-03-01',
        '8_ago_quantity': '2021-04-01',
        '7_ago_quantity': '2021-05-01',
        '6_ago_quantity': '2021-06-01',
        '5_ago_quantity': '2021-07-01',
        '4_ago_quantity': '2021-08-01',
        '3_ago_quantity': '2021-09-01',
        '2_ago_quantity': '2021-10-01',
        '1_ago_quantity': '2021-11-01'
    }, inplace=True)
    # 条件查询pandas数据
    # df = df[df['product_create_time'] < '2020-12-01']
    df = df.reset_index(drop=True)
    # 将空值转换为0
    df[date_col] = df[date_col].fillna(0)
    print(len(df))
    ishead = True
    for i in range(0, len(df)):
        sku = df.at[i, 'sku_code']
        dff = df.loc[[i]].rolling(window=3, min_periods=3, axis=1).mean()
        dff = dff.reindex(columns=list(col2), fill_value=sku)
        dff = dff.fillna(0)
        # print(dff)
        if ishead:
            csv_writer.writerow(dff)
            ishead = False
        csv_writer.writerow(dff.values.tolist()[0])
        csv_wri.writerow(df.loc[[i]].values.tolist()[0])
    print(len(df))
    f.close()
    yy.close()
    print("dff")

def AriMa_MA2():
    ari = open('arima_maToHive.csv', 'w', encoding='utf-8', newline='')
    wei = open('yuan.csv', 'w', encoding='utf-8', newline='')
    colnum = ['sku_code', '12_ago', '11_ago', '10_ago', '9_ago', '8_ago', '7_ago', '6_ago', '5_ago', '4_ago', '3_ago', '2_ago', '1_ago', 'm_0_fc', 'm_1_fc', 'm_2_fc', 'm_3_fc', 'm_4_fc', 'm_5_fc', 'm_6_fc', 'forecast_date']
    arima_csv = csv.writer(ari)
    yuan = csv.writer(wei)
    # 移动平均值
    df = pd.read_csv('MA.csv')
    # 真实值
    ddf = pd.read_csv('M.csv')
    dflist = df.values.tolist()
    ddflist = ddf.values.tolist()
    sku = ''
    isHead = True
    time1 = time.strftime('%Y%m%d', time.localtime(time.time()))
    time1 = "20211202"
    str_sql = ""
    str_sql = "insert overwrite table ehsy.ods_pms_sku_forecast_arima partition (pt='" \
                      + time1 + "') values " + str_sql
    jishu = 0
    for num in range(0, len(ddflist)):
        sku = ddflist[num][0]
        dds = pd.Series(ddflist[num][1:10])
        py = pyads.cursor()
        print(jishu)
        dds.index = pd.Index(sm.tsa.datetools.dates_from_range('2020m12', '2021m8'))
        #做一阶差分，增加数据平稳性
        # dfsw = ds.diff()
        # dfsw = dfsw.dropna()
        # # plt.plot(dfsw)
        # # # 二阶
        # # dfsw = dfsw.diff()
        # # plt.plot(dfsw)
        # # plt.title('二阶差分')
        # # plt.show()
        # # dfsw = dfsw.dropna()
        # # plt.figure()
        #
        # acf = plot_acf(dfsw, lags=8)
        # pacf = plot_pacf(dfsw, lags=3)


        model = ARIMA(dds, order=(1, 1, 1))
        result = model.fit()
        # 统计出ARIMA模型的指标
        result.summary()
        # 预测，指定起始与终止时间。预测值起始时间必须在原始数据中，终止时间不需要
        pred = result.predict('20210831', '20220331', dynamic=True, typ='levels')
        if isHead:
            arima_csv.writerow(colnum)
            isHead = False
        p = pred.values.tolist()
        yuan = [ddflist[num][10], ddflist[num][11], ddflist[num][12]]
        yuce = p[0:3]
        RMSE = metrics.mean_squared_error(yuan, yuce) ** 0.5
        print(RMSE)
        str_sql = str_sql+str(tuple([sku]+ddflist[num][1:]+[p[0], p[1], p[2], p[3], p[4], p[5], p[6], time1, RMSE]))+","

        jishu += 1
        if jishu >= 2000:
            str_sql = str_sql[0:len(str_sql)-1]
            py.execute(str_sql)
            str_sql = ""
            str_sql = "insert into table ehsy.ods_pms_sku_forecast_arima partition (pt='"+time1+"') values " + str_sql
            jishu = 0
    if len(str_sql) > 100:
        str_sql = str_sql[0:len(str_sql) - 1]
        py.execute(str_sql)
    ari.close()
    wei.close()

if __name__ == '__main__':
    # 可以显示中文
    getMa()
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    AriMa_MA2()

--------------arima2--------------------
# arima处理数据到Hive

import datetime
import time

from sklearn import metrics
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
from impala.dbapi import connect
import matplotlib.pyplot as plt
import csv

pyads = connect('hiveserver2.ehsy.com', 10000, 'ehsy', auth_mechanism='PLAIN', user='readonly', password='Ehsyol@2021')

#
#

def getMa():
    f = open('MA.csv', 'w', encoding='utf-8', newline='')
    yy = open('M.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_wri = csv.writer(yy)
    date_col = ['2020-12-01', '2021-01-01', '2021-02-01',
                '2021-03-01', '2021-04-01', '2021-05-01',
                '2021-06-01', '2021-07-01', '2021-08-01',
                '2021-09-01', '2021-10-01', '2021-11-01']
    # 列名
    col = ['sku_code','2020-12-01', '2021-01-01', '2021-02-01',
           '2021-03-01', '2021-04-01', '2021-05-01',
           '2021-06-01', '2021-07-01', '2021-08-01',
           '2021-09-01', '2021-10-01', '2021-11-01']
    col2 = ['sku_code', '2020-12-01', '2021-01-01', '2021-02-01',
            '2021-03-01', '2021-04-01', '2021-05-01',
            '2021-06-01', '2021-07-01', '2021-08-01',
            '2021-09-01', '2021-10-01', '2021-11-01']
    py = pyads.cursor()

    sql = "select sku_code,12_ago_quantity,11_ago_quantity,10_ago_quantity,9_ago_quantity" \
          ",8_ago_quantity ,7_ago_quantity,6_ago_quantity,5_ago_quantity,4_ago_quantity" \
          ",3_ago_quantity,2_ago_quantity ,1_ago_quantity " \
          "from ehsy.ods_bh_sale_report_stats_country_lv1  where model='ARIMA'"
    print(sql)
    py.execute(sql)
    # 将py_hive查询的数据给到pandas
    df = pd.DataFrame(py.fetchall(), columns=col)
    # 重命名列名
    df.rename(columns={
        '12_ago_quantity': '2020-12-01',
        '11_ago_quantity': '2021-01-01',
        '10_ago_quantity': '2021-02-01',
        '9_ago_quantity': '2021-03-01',
        '8_ago_quantity': '2021-04-01',
        '7_ago_quantity': '2021-05-01',
        '6_ago_quantity': '2021-06-01',
        '5_ago_quantity': '2021-07-01',
        '4_ago_quantity': '2021-08-01',
        '3_ago_quantity': '2021-09-01',
        '2_ago_quantity': '2021-10-01',
        '1_ago_quantity': '2021-11-01'
    }, inplace=True)
    # 条件查询pandas数据
    # df = df[df['product_create_time'] < '2020-12-01']
    df = df.reset_index(drop=True)
    # 将空值转换为0
    df[date_col] = df[date_col].fillna(0)
    print(len(df))
    ishead = True
    for i in range(0, len(df)):
        sku = df.at[i, 'sku_code']
        dff = df.loc[[i]].rolling(window=3, min_periods=3, axis=1).mean()
        dff = dff.reindex(columns=list(col2), fill_value=sku)
        dff = dff.fillna(0)
        # print(dff)
        if ishead:
            csv_writer.writerow(dff)
            ishead = False
        csv_writer.writerow(dff.values.tolist()[0])
        csv_wri.writerow(df.loc[[i]].values.tolist()[0])
    print(len(df))
    f.close()
    yy.close()
    print("dff")

def AriMa_MA2():
    ari = open('arima_maToHive.csv', 'w', encoding='utf-8', newline='')
    wei = open('yuan.csv', 'w', encoding='utf-8', newline='')
    colnum = ['sku_code', '12_ago', '11_ago', '10_ago', '9_ago', '8_ago', '7_ago', '6_ago', '5_ago', '4_ago', '3_ago', '2_ago', '1_ago', 'm_0_fc', 'm_1_fc', 'm_2_fc', 'm_3_fc', 'm_4_fc', 'm_5_fc', 'm_6_fc', 'forecast_date']
    arima_csv = csv.writer(ari)
    yuan = csv.writer(wei)
    # 移动平均值
    df = pd.read_csv('MA.csv')
    # 真实值
    ddf = pd.read_csv('M.csv')
    dflist = df.values.tolist()
    ddflist = ddf.values.tolist()
    sku = ''
    isHead = True
    time1 = time.strftime('%Y%m%d', time.localtime(time.time()))
    time1 = '20211203'
    str_sql = ""
    str_sql = "insert overwrite table ehsy.ods_pms_sku_forecast_arima partition (pt='"+time1 + "') values " + str_sql
    jishu = 0
    py = pyads.cursor()
    for num in range(0, len(ddflist)):
        sku = ddflist[num][0]
        dds = pd.Series(ddflist[num][1:13])
        print(dds)
        dds.index = pd.Index(sm.tsa.datetools.dates_from_range('2020m12', '2021m11'))
        #做一阶差分，增加数据平稳性
        # dfsw = ds.diff()
        # dfsw = dfsw.dropna()
        # # plt.plot(dfsw)
        # # # 二阶
        # # dfsw = dfsw.diff()
        # # plt.plot(dfsw)
        # # plt.title('二阶差分')
        # # plt.show()
        # # dfsw = dfsw.dropna()
        # # plt.figure()
        #
        # acf = plot_acf(dfsw, lags=8)
        # pacf = plot_pacf(dfsw, lags=3)


        model = ARIMA(dds, order=(1, 1, 1))
        result = model.fit()
        # 统计出ARIMA模型的指标
        result.summary()
        # 预测，指定起始与终止时间。预测值起始时间必须在原始数据中，终止时间不需要
        pred = result.predict('20211130', '20220731', dynamic=True, typ='levels')
        if isHead:
            arima_csv.writerow(colnum)
            isHead = False
        p = pred.values.tolist()

        # yuan = [ddflist[num][10], ddflist[num][11], ddflist[num][12]]
        # yuce = p[0:3]
        # RMSE = metrics.mean_squared_error(yuan, yuce) ** 0.5
        # print(RMSE)

        str_sql = str_sql+str(tuple([sku]+ddflist[num][1:]+[p[0], p[1], p[2], p[3], p[4], p[5], p[6], time1, 0.0]))+","
        print(str_sql)
        jishu += 1
        if jishu >= 2000:
            str_sql = str_sql[0:len(str_sql)-1]
            py.execute(str_sql)
            str_sql = ""
            str_sql = "insert into table ehsy.ods_pms_sku_forecast_arima partition (pt='"+time1+"') values " + str_sql
            jishu = 0
    if len(str_sql) > 100:
        str_sql = str_sql[0:len(str_sql) - 1]
        py.execute(str_sql)
    ari.close()
    wei.close()

if __name__ == '__main__':
    # 可以显示中文
    getMa()
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    AriMa_MA2()


---------------------arima3---------------------------
# arima处理数据到Hive

import datetime
import time

from sklearn import metrics
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
from impala.dbapi import connect
import matplotlib.pyplot as plt
import csv

pyads = connect('hiveserver2.ehsy.com', 10000, 'ehsy', auth_mechanism='PLAIN', user='readonly', password='Ehsyol@2021')

#
#

def getMa():
    f = open('MA.csv', 'w', encoding='utf-8', newline='')
    yy = open('M.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_wri = csv.writer(yy)
    date_col = ['2020-12-01', '2021-01-01', '2021-02-01',
                '2021-03-01', '2021-04-01', '2021-05-01',
                '2021-06-01', '2021-07-01', '2021-08-01',
                '2021-09-01', '2021-10-01', '2021-11-01']
    # 列名
    col = ['sku_code','2020-12-01', '2021-01-01', '2021-02-01',
           '2021-03-01', '2021-04-01', '2021-05-01',
           '2021-06-01', '2021-07-01', '2021-08-01',
           '2021-09-01', '2021-10-01', '2021-11-01']
    col2 = ['sku_code', '2020-12-01', '2021-01-01', '2021-02-01',
            '2021-03-01', '2021-04-01', '2021-05-01',
            '2021-06-01', '2021-07-01', '2021-08-01',
            '2021-09-01', '2021-10-01', '2021-11-01']
    py = pyads.cursor()

    sql = "select sku_code,12_ago_quantity,11_ago_quantity,10_ago_quantity,9_ago_quantity" \
          ",8_ago_quantity ,7_ago_quantity,6_ago_quantity,5_ago_quantity,4_ago_quantity" \
          ",3_ago_quantity,2_ago_quantity ,1_ago_quantity " \
          "from ehsy.ods_bh_sale_report_stats_country_lv1 "
    print(sql)
    py.execute(sql)
    # 将py_hive查询的数据给到pandas
    df = pd.DataFrame(py.fetchall(), columns=col)
    # 重命名列名
    df.rename(columns={
        '12_ago_quantity': '2020-12-01',
        '11_ago_quantity': '2021-01-01',
        '10_ago_quantity': '2021-02-01',
        '9_ago_quantity': '2021-03-01',
        '8_ago_quantity': '2021-04-01',
        '7_ago_quantity': '2021-05-01',
        '6_ago_quantity': '2021-06-01',
        '5_ago_quantity': '2021-07-01',
        '4_ago_quantity': '2021-08-01',
        '3_ago_quantity': '2021-09-01',
        '2_ago_quantity': '2021-10-01',
        '1_ago_quantity': '2021-11-01'
    }, inplace=True)
    # 条件查询pandas数据
    # df = df[df['product_create_time'] < '2020-12-01']
    df = df.reset_index(drop=True)
    # 将空值转换为0
    df[date_col] = df[date_col].fillna(0)
    print(len(df))
    ishead = True
    for i in range(0, len(df)):
        sku = df.at[i, 'sku_code']
        dff = df.loc[[i]].rolling(window=3, min_periods=3, axis=1).mean()
        dff = dff.reindex(columns=list(col2), fill_value=sku)
        dff = dff.fillna(0)
        # print(dff)
        if ishead:
            csv_writer.writerow(dff)
            ishead = False
            csv_wri.writerow(dff)
        csv_writer.writerow(dff.values.tolist()[0])
        csv_wri.writerow(df.loc[[i]].values.tolist()[0])
    print(len(df))
    f.close()
    yy.close()
    print("dff")

def AriMa_MA2():
    ari = open('arima_maToHive.csv', 'w', encoding='utf-8', newline='')
    wei = open('yuan.csv', 'w', encoding='utf-8', newline='')
    colnum = ['sku_code', '12_ago', '11_ago', '10_ago', '9_ago', '8_ago', '7_ago', '6_ago', '5_ago', '4_ago', '3_ago', '2_ago', '1_ago', 'm_0_fc', 'm_1_fc', 'm_2_fc', 'm_3_fc', 'm_4_fc', 'm_5_fc', 'm_6_fc', 'forecast_date']
    arima_csv = csv.writer(ari)
    yuan = csv.writer(wei)
    # 移动平均值
    df = pd.read_csv('MA.csv')
    # 真实值
    ddf = pd.read_csv('M.csv')
    dflist = df.values.tolist()
    ddflist = ddf.values.tolist()
    sku = ''
    isHead = True
    time1 = time.strftime('%Y%m%d', time.localtime(time.time()))
    time1 = "20211229"
    str_sql = ""
    str_sql = "insert overwrite table ehsy.ods_pms_sku_forecast_arima partition (pt='" \
                      + time1 + "') values " + str_sql
    jishu = 0
    for num in range(0, len(ddflist)):
        sku = ddflist[num][0]
        dds = pd.Series(ddflist[num][1:10])
        py = pyads.cursor()
        print(jishu)
        dds.index = pd.Index(sm.tsa.datetools.dates_from_range('2020m12', '2021m8'))
        #做一阶差分，增加数据平稳性
        # dfsw = ds.diff()
        # dfsw = dfsw.dropna()
        # # plt.plot(dfsw)
        # # # 二阶
        # # dfsw = dfsw.diff()
        # # plt.plot(dfsw)
        # # plt.title('二阶差分')
        # # plt.show()
        # # dfsw = dfsw.dropna()
        # # plt.figure()
        #
        # acf = plot_acf(dfsw, lags=8)
        # pacf = plot_pacf(dfsw, lags=3)


        model = ARIMA(dds, order=(1, 1, 3))
        result = model.fit()
        # 统计出ARIMA模型的指标
        result.summary()
        # 预测，指定起始与终止时间。预测值起始时间必须在原始数据中，终止时间不需要
        pred = result.predict('20210831', '20220331', dynamic=True, typ='levels')
        if isHead:
            arima_csv.writerow(colnum)
            isHead = False
        p = pred.values.tolist()
        yuan = [ddflist[num][10], ddflist[num][11], ddflist[num][12]]
        yuce = p[0:3]
        RMSE = metrics.mean_squared_error(yuan, yuce) ** 0.5
        print(RMSE)
        str_sql = str_sql+str(tuple([sku]+ddflist[num][1:]+[p[0], p[1], p[2], p[3], p[4], p[5], p[6], time1, RMSE, '']))+","

        jishu += 1
        if jishu >= 2000:
            str_sql = str_sql[0:len(str_sql)-1]
            py.execute(str_sql)
            str_sql = ""
            str_sql = "insert into table ehsy.ods_pms_sku_forecast_arima partition (pt='"+time1+"') values " + str_sql
            jishu = 0
    if len(str_sql) > 100:
        str_sql = str_sql[0:len(str_sql) - 1]
        py.execute(str_sql)
    ari.close()
    wei.close()

if __name__ == '__main__':
    # 可以显示中文
    getMa()
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    AriMa_MA2()


------------crston------------------------------------
import time

import numpy as np
import pandas as pd
from impala.dbapi import connect
from scipy import stats

# alpha 平滑指数
def Croston(ts, extra_periods=1, alpha=0.5):
    d = np.array(ts)  # Transform the input into a numpy array
    cols = len(d)  # Historical period length
    d = np.append(d, [np.nan] * extra_periods)  # Append np.nan into the demand array to cover future periods

    # level (a), periodicity(p) and forecast (f)
    a, p, f = np.full((3, cols + extra_periods), np.nan)
    q = 1  # periods since last demand observation

    # Initialization
    first_occurence = np.argmax(d[:cols] > 0)
    a[0] = d[first_occurence]
    p[0] = 1 + first_occurence
    f[0] = a[0] / p[0]
    # Create all the t+1 forecasts
    for t in range(0, cols):
        if d[t] > 0:
            a[t + 1] = alpha * d[t] + (1 - alpha) * a[t]
            p[t + 1] = alpha * q + (1 - alpha) * p[t]
            f[t + 1] = a[t + 1] / p[t + 1]
            q = 1
        else:
            a[t + 1] = a[t]
            p[t + 1] = p[t]
            f[t + 1] = f[t]
            q += 1
    # Future Forecast
    a[cols + 1:cols + extra_periods] = a[cols]
    p[cols + 1:cols + extra_periods] = p[cols]
    f[cols + 1:cols + extra_periods] = f[cols]
    df = pd.DataFrame.from_dict({"Demand": d, "Forecast": f, "Period": p, "Level": a, "Error": d - f})
    return df


def Croston_TSB(ts, extra_periods=1, alpha=0.4, beta=0.6):
    d = np.array(ts)  # Transform the input into a numpy array
    cols = len(d)  # Historical period length
    d = np.append(d, [np.nan] * extra_periods)  # Append np.nan into the demand array to cover future periods

    # level (a), probability(p) and forecast (f)
    a, p, f = np.full((3, cols + extra_periods), np.nan)
    # Initialization
    first_occurence = np.argmax(d[:cols] > 0)
    a[0] = d[first_occurence]
    p[0] = 1 / (1 + first_occurence)
    f[0] = p[0] * a[0]

    # Create all the t+1 forecasts
    for t in range(0, cols):
        if d[t] > 0:
            a[t + 1] = alpha * d[t] + (1 - alpha) * a[t]
            p[t + 1] = beta * (1) + (1 - beta) * p[t]
        else:
            a[t + 1] = a[t]
            p[t + 1] = (1 - beta) * p[t]
            f[t + 1] = p[t + 1] * a[t + 1]

    # Future Forecast
    a[cols + 1:cols + extra_periods] = a[cols]
    p[cols + 1:cols + extra_periods] = p[cols]
    f[cols + 1:cols + extra_periods] = f[cols]
    df = pd.DataFrame.from_dict({"Demand": d, "Forecast": f, "Period": p, "Level": a, "Error": d - f})
    return df

PORT = 10000
HOST = 'hiveserver2.ehsy.com'
conn = connect('hiveserver2.ehsy.com', 10000, 'ehsy', auth_mechanism='PLAIN', user='readonly', password='Ehsyol@2021')


print("连接成功")
cursor = conn.cursor()
cursor.execute("select sku_code,cov,12_ago_quantity,11_ago_quantity,10_ago_quantity,9_ago_quantity"
               ",8_ago_quantity ,7_ago_quantity,6_ago_quantity,5_ago_quantity,4_ago_quantity"
               ",3_ago_quantity,2_ago_quantity ,1_ago_quantity "
               "from ehsy.ods_bh_sale_report_stats_country_lv1 where model =  'CROSTON'")
print("查询成功")
date_col = ['sku_code','cov', '12_ago_quantity', '11_ago_quantity',
            '10_ago_quantity', '9_ago_quantity', '8_ago_quantity',
            '7_ago_quantity', '6_ago_quantity', '5_ago_quantity',
            '4_ago_quantity', '3_ago_quantity', '2_ago_quantity',
            '1_ago_quantity']
quality_col = ['12_ago_quantity', '11_ago_quantity',
               '10_ago_quantity', '9_ago_quantity', '8_ago_quantity',
               '7_ago_quantity', '6_ago_quantity', '5_ago_quantity',
               '4_ago_quantity', '3_ago_quantity', '2_ago_quantity',
               '1_ago_quantity']
calc_col = ['12_ago_quantity', '11_ago_quantity',
            '10_ago_quantity', '9_ago_quantity', '8_ago_quantity',
            '7_ago_quantity', '6_ago_quantity', '5_ago_quantity',
            '4_ago_quantity', '3_ago_quantity', '2_ago_quantity',
            '1_ago_quantity', 'max', 'std']
df = pd.DataFrame(cursor.fetchall(), columns=date_col)
print("df")
def Croston_TSB_Cal(x):
    tsb_f = Croston_TSB(x)
    return tsb_f.values[-1][1]

def Croston_Cal(x):
    tsb_f = Croston(x)
    return tsb_f.values[-1][1]


quality_col1 = ['12_ago_quantity', '11_ago_quantity',
                '10_ago_quantity', '9_ago_quantity', '8_ago_quantity',
                '7_ago_quantity', '6_ago_quantity', '5_ago_quantity',
                '4_ago_quantity','3_ago_quantity', '2_ago_quantity',
                '1_ago_quantity', 'croston_f0']
quality_col2 = ['12_ago_quantity', '11_ago_quantity',
                '10_ago_quantity', '9_ago_quantity', '8_ago_quantity',
                '7_ago_quantity', '6_ago_quantity', '5_ago_quantity',
                '4_ago_quantity', '3_ago_quantity', '2_ago_quantity',
                '1_ago_quantity', 'croston_f0', 'croston_f1']
quality_col3 = ['12_ago_quantity', '11_ago_quantity',
                '10_ago_quantity', '9_ago_quantity', '8_ago_quantity',
                '7_ago_quantity', '6_ago_quantity', '5_ago_quantity',
                '4_ago_quantity','3_ago_quantity', '2_ago_quantity',
                '1_ago_quantity', 'croston_f0', 'croston_f1', 'croston_f2']
quality_col4 = ['12_ago_quantity', '11_ago_quantity',
                '10_ago_quantity', '9_ago_quantity', '8_ago_quantity',
                '7_ago_quantity', '6_ago_quantity', '5_ago_quantity',
                '4_ago_quantity', '3_ago_quantity', '2_ago_quantity',
                '1_ago_quantity', 'croston_f0', 'croston_f1', 'croston_f2', 'croston_f3']
quality_col5 = ['12_ago_quantity', '11_ago_quantity',
                '10_ago_quantity', '9_ago_quantity', '8_ago_quantity',
                '7_ago_quantity', '6_ago_quantity', '5_ago_quantity',
                '4_ago_quantity','3_ago_quantity', '2_ago_quantity',
                '1_ago_quantity', 'croston_f0', 'croston_f1', 'croston_f2', 'croston_f3', 'croston_f4']
quality_col6 = ['12_ago_quantity', '11_ago_quantity',
                '10_ago_quantity', '9_ago_quantity', '8_ago_quantity',
                '7_ago_quantity', '6_ago_quantity', '5_ago_quantity',
                '4_ago_quantity','3_ago_quantity', '2_ago_quantity',
                '1_ago_quantity', 'croston_f0', 'croston_f1', 'croston_f2', 'croston_f3', 'croston_f4', 'croston_f5']

quality_3 = ['12_ago_quantity', '11_ago_quantity',
             '10_ago_quantity', '9_ago_quantity', '8_ago_quantity',
             '7_ago_quantity', '6_ago_quantity', '5_ago_quantity',
             '4_ago_quantity']
quality_2 = ['12_ago_quantity', '11_ago_quantity',
             '10_ago_quantity', '9_ago_quantity', '8_ago_quantity',
             '7_ago_quantity', '6_ago_quantity', '5_ago_quantity',
             '4_ago_quantity', '3_ago_quantity']

quality_1 = ['12_ago_quantity', '11_ago_quantity',
             '10_ago_quantity', '9_ago_quantity', '8_ago_quantity',
             '7_ago_quantity', '6_ago_quantity', '5_ago_quantity',
             '4_ago_quantity', '3_ago_quantity', '2_ago_quantity']


# RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# 前3月销量
df['croston_3'] = df[quality_3].apply(lambda x: Croston_Cal(x), axis=1)
# 前2月销量
df['croston_2'] = df[quality_2].apply(lambda x: Croston_Cal(x), axis=1)
# 前1月销量
df['croston_1'] = df[quality_1].apply(lambda x: Croston_Cal(x), axis=1)


# MAPE
def mape(y_true, y_pred):
    for v in y_true:
        if v == 0:
            return np.nan
    return np.mean(np.abs((y_pred - y_true) / y_true))

# R-SQUAT
def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    #a、b、r
    return r_value**2


# 当月销量
df['croston_f0'] = df[quality_col].apply(lambda x: Croston_Cal(x), axis=1)
# 下月销量
df['croston_f1'] = df[quality_col1].apply(lambda x: Croston_Cal(x), axis=1)
# 下下月销量
df['croston_f2'] = df[quality_col2].apply(lambda x: Croston_Cal(x), axis=1)

df['croston_f3'] = df[quality_col3].apply(lambda x: Croston_Cal(x), axis=1)

df['croston_f4'] = df[quality_col4].apply(lambda x: Croston_Cal(x), axis=1)

df['croston_f5'] = df[quality_col5].apply(lambda x: Croston_Cal(x), axis=1)

df['croston_f6'] = df[quality_col6].apply(lambda x: Croston_Cal(x), axis=1)

# df['RMSE'] = df[['3_ago_quantity', '2_ago_quantity', '1_ago_quantity', 'croston_3', 'croston_2', 'croston_1']]\
#     .apply(lambda x: rmse(np.array(x[['3_ago_quantity', '2_ago_quantity', '1_ago_quantity']])
#                           , np.array(x[['croston_3', 'croston_2', 'croston_1']])), axis=1)
# df['MAPE'] = df[['3_ago_quantity', '2_ago_quantity', '1_ago_quantity', 'croston_3', 'croston_2', 'croston_1']]\
#     .apply(lambda x: mape(np.array(x[['3_ago_quantity', '2_ago_quantity', '1_ago_quantity']])
#                           , np.array(x[['croston_3', 'croston_2', 'croston_1']])), axis=1)
# df['R-SQUAT'] = df[['3_ago_quantity', '2_ago_quantity', '1_ago_quantity', 'croston_3', 'croston_2', 'croston_1']]\
#     .apply(lambda x: rsquared(np.array(x[['3_ago_quantity', '2_ago_quantity', '1_ago_quantity']])
#                               , np.array(x[['croston_3', 'croston_2', 'croston_1']])), axis=1)
print("计算完成")
df.to_csv("croston_cov.csv")
df.reset_index()
df = df.values.tolist()
print("toList")
str_sql = ""
str_sql = "insert into table ehsy.ods_pms_sku_forecast_croston partition (pt='" \
          + time.strftime('%Y%m%d', time.localtime(time.time())) + "') values " + str_sql
jishu = 0
time1 = time.strftime('%Y%m%d', time.localtime(time.time()))
for num in range(0, len(df)):
    print(num)
    str_sql = str_sql + str(tuple([df[num][0], df[num][2], df[num][3], df[num][4], df[num][5], df[num][6], df[num][7], df[num][8], df[num][9], df[num][10], df[num][11], df[num][12], df[num][13], df[num][17], df[num][18], df[num][19], df[num][20], df[num][21], df[num][22], df[num][23], time1])) + ","
    if jishu >= 2000:
        str_sql = str_sql[0:len(str_sql)-1]
        cursor.execute(str_sql)
       # print(str_sql)
        str_sql = ""
        str_sql = "insert into table ehsy.ods_pms_sku_forecast_croston partition (pt='" \
                  + time.strftime('%Y%m%d', time.localtime(time.time())) + "') values " + str_sql
        jishu = 0
    jishu += 1

if len(str_sql) >= 100:
    str_sql = str_sql[0:len(str_sql) - 1]
    cursor.execute(str_sql)
   # print(str_sql)
    print('完成')





