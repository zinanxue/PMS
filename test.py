import customise as cust
import pandas as pd

# get data
cursor = cust.pyhive_connect()
sql = cust.select_clause()
cursor.execute(sql)

# get date columns
col = cust.recent_date_col(12)
col.insert(0, 'sku_code')

# transfer the data into time series
df = pd.DataFrame(cursor.fetchall(), columns=col)
df.set_index('sku_code', inplace=True)
df = df.stack()
df.index = df.index.rename('date_col', level=1)
df.name = 'sale_amount'
df = df.reset_index()
# use to fetch specific sku
df = df.set_index('sku_code')


class GetSkuMa():
    def __init__(self, df, date_col_name, amount_col_name):
        self.df = df
        self.date_col_name = date_col_name
        self.amount_col_name = amount_col_name

    def get_sku(self):
        self.sku = input('输入一个sku_code:')
        return self.df.loc[self.sku, :]

    def get_ma(self):
        window = int(input('输入window大小：'))
        min_periods = int(input('输入min_periods大小：'))
        ma_df = self.df.copy()
        ma_df.loc[:, f'ma{window}'] = self.df[self.amount_col_name].rolling(window=window, min_periods=min_periods
                                                                            ).mean()
        return ma_df



sku = GetSkuMa(df, 'date_col', 'sale_amount')
df_sku = sku.get_sku()
df_sku_ma = sku.get_ma()


