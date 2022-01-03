from pyhive import hive
# from impala.dbapi import connect
import time


def pyhive_connect():
    PORT = 10000
    HOST = 'hiveserver2.ehsy.com'
    conn = hive.Connection(host=HOST, port=PORT, database='ehsy')
    return conn.cursor()


def impyla_conncet():
    conn = connect('hiveserver2.ehsy.com', 10000, 'ehsy', auth_mechanism='PLAIN', user='readonly',
                   password='Ehsyol@2021')
    return conn.cursor()


def select_clause():
    sql = '''
        select sku_code, 12_ago_quantity, 11_ago_quantity, 10_ago_quantity,
            9_ago_quantity, 8_ago_quantity, 7_ago_quantity, 6_ago_quantity, 
            5_ago_quantity, 4_ago_quantity, 3_ago_quantity, 2_ago_quantity,
            1_ago_quantity
        from ehsy.ods_bh_sale_report_stats_country_lv1
        
    '''
    return sql


def recent_date_col(n: int):
    now = time.localtime()
    date_col = []
    time_col = [time.localtime(time.mktime((now.tm_year, now.tm_mon - i, 1, 0, 0, 0, 0, 0, 0)))[:2]
                for i in range(1, n + 1)]
    for month in time_col:
        if month[1] <=9:
            date_col.append(str(month[0]) + '-0' + str(month[1]))
        elif month[1] > 9:
            date_col. append(str(month[0]) + '-' + str(month[1]))
        else:
            return 'error'
    return date_col


def main():
    sql = select_clause()
    cursor = pyhive_connect()
    cursor.execute(sql)
    date_col = recent_date_col(12)
    print(cursor.fetchall())


if __name__ == '__main__':
    # main()
    print(recent_date_col(60))

# df1['开票日期'] = df1.开票日期.apply(lambda x: x.strftime('%Y-%m')).astype('datetime64') # 先按想要的格式转为字符串，再转为日期格式

