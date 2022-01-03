import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from impala.dbapi import connect

conn = connect('10.80.66.16', 10000, 'ehsy', auth_mechanism='PLAIN', user='karon_xue', password='karon_xue@ehsy27')
cursor = conn.cursor()
cursor.execute("select sku_code,12_ago_quantity,11_ago_quantity,10_ago_quantity,"
               "9_ago_quantity,8_ago_quantity ,7_ago_quantity,6_ago_quantity,"
               "5_ago_quantity,4_ago_quantity,3_ago_quantity,2_ago_quantity ,"
               "1_ago_quantity,product_create_time "
               "from ehsy.ads_bh_sale_report_stats_country "
               "where available_type =  '符合备货逻辑' limit 20")

# 数据源变动所用到的col也要变
col = ['sku_code', '2020-12-01', '2021-01-01', '2021-02-01',
       '2021-03-01', '2021-04-01', '2021-05-01',
       '2021-06-01', '2021-07-01', '2021-08-01',
       '2021-09-01', '2021-10-01', '2021-11-01', 'product_create_time']
# 数值所在列
date_col = ['2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01',
            '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01',
            '2021-09-01', '2021-10-01', '2021-11-01']

df = pd.DataFrame(cursor.fetchall(), columns=col)

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

df1 = df[col]
df1.set_index('sku_code', inplace=True)  # 设置索引

sku_init = df1.index[0]  # 初始化第一个sku_code
df_init = df1.loc[sku_init, date_col].reset_index()  # 获得新的index，原来的index(sku_code)变成数据列，保留下来。
df_init.columns = ['time', 'sum']  # 重置列名
df_init['sku_code'] = sku_init  # 将初始化sku_code赋值给一个新列

# 将剩余sku_code循环以上操作
for i in range(1, len(df1.index)):
    sku = df1.index[i]
    temp = df1.loc[sku, date_col].reset_index()
    temp.columns = ['time', 'sum']
    temp['sku_code'] = sku
    df_init = pd.concat((df_init, temp), axis=0)  # 将格式化好的sku_code与初始化的sku_code数据拼接


# 如果要输出所有sku的结果执行下面
# csv_name = '' #填充名字
# df_init.to_csv(f'{csv_name}.csv')

# 自定义
class GetSkuMa:

    def __init__(self, df):
        self.df = df

    def get_sku(self, sku_col_name):
        self.sku_col_name = sku_col_name
        self.sku = input('输入一个sku_code:')

        return self.df[self.df[self.sku_col_name] == self.sku]

    def get_ma(self, sum_col_name):
        self.sum_col_name = sum_col_name
        self.window = int(input('输入window大小:'))
        self.min_periods = int(input('输入min_periods大小:'))

        self.df.loc[:, f'ma{window}'] = self.df[sum_col_name].rolling(window=window, min_periods=min_periods).mean()

        return self.df


sku = GetSkuMa(df_init)
df_sku = sku.get_sku('sku_code')  # 返回一个sku的数据

df_sku_ma = GetSkuMa(df_sku).get_ma('sum')  # 返回那个sku的数据和ma数据
