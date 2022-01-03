from pyhive import hive


# import pandas as pd

# import seaborn as sns
# import matplotlib.pyplot as plt
# from impala.dbapi import connect
# from impala.dbapi import connect

def hh():
    PORT = 10000
    HOST = 'hiveserver2.ehsy.com'
    conn = hive.Connection(host=HOST, port=PORT, database='ehsy')
    # conn = connect('hiveserver2.ehsy.com', 10000, 'ehsy', auth_mechanism='PLAIN', user='readonly', password='Ehsyol@2021')
    cursor = conn.cursor()
    cursor.execute("select sku_code,12_ago_quantity,11_ago_quantity,10_ago_quantity,9_ago_quantity"
                   ",8_ago_quantity ,7_ago_quantity,6_ago_quantity,5_ago_quantity,4_ago_quantity"
                   ",3_ago_quantity,2_ago_quantity ,1_ago_quantity,product_create_time "
                   "from ehsy.ads_bh_sale_report_stats_country where available_type =  '符合备货逻辑'"
                   "limit 202")
    print(cursor.fetchall())


if __name__ == '__main__':
    hh()
