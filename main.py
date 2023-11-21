import pandas as pd
from pandasql import sqldf
import numpy as np
import sqlite3
from intervaltree import IntervalTree


def load_data(url):
    data = pd.read_csv(url, parse_dates=['signup_time', 'purchase_time'],
                       converters={'ip_address': lambda x: preprocess_ip_to_int(x) if pd.notnull(x) else x})
    # converters={'ip_address': lambda x: convert_to_int(x)})
    return data.copy()


def load_ip(url):
    ip = pd.read_csv(url, sep=';')
    return ip.copy()


# check the datetime format if correct formated in all the raws on both time columns
def check_datetime_format(data):
    check_format = '%Y-%m-%d %H:%M:%S'
    columns_to_check = ['signup_time', 'purchase_time']
    incorrect_format_column = [col for col in columns_to_check if
                               not pd.to_datetime(data[col], format=check_format, errors='coerce').notnull().all()]
    if not incorrect_format_column:
        return "Datetime is correctly formatted in the both columns"
    else:
        return f"Datetime format is not correct in columns: {', '.join(incorrect_format_column)}"


def convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return -1


def preprocess_ip_to_int(ip):
    return int(ip.split('.')[0])


# take too much time to perform
# def replace_Ip_country(data_fraud, data_ip):
#     query = """
#         SELECT *
#         FROM data_fraud df
#         LEFT JOIN data_ip dip
#         ON df.ip_address >= dip.lower_bound_ip_address AND df.ip_address <= dip.upper_bound_ip_address;
#     """
#     result = sqldf(query, locals())
#     return result.copy()

# def replace_Ip_country(data_fraud, data_ip):
#     connetion = sqlite3.connect('fraud_detection.db')
#     data_fraud.to_sql('data_fraud', connetion, index=False)
#     data_ip.to_sql('data_ip', connetion, index=False)
#     connetion.execute('CREATE INDEX lower_idx ON data_ip(lower_bound_ip_address);')
#     connetion.execute('CREATE INDEX upper_idx ON data_ip(upper_bound_ip_address);')
#
#     query = """
#         SELECT *
#         FROM data_fraud df
#         LEFT JOIN data_ip dip
#         ON df.ip_address >= dip.lower_bound_ip_address
#         AND df.ip_address <= dip.upper_bound_ip_address;
#     """
#
#     result = pd.read_sql_query(query, connetion)
#     connetion.close()
#     return result

def replace_Ip_country(data_fraud, data_ip):
    search_tree = IntervalTree()
    for row in data_ip.itertuples(index=False):
        search_tree[row.lower_bound_ip_address:row.upper_bound_ip_address] = row.country

    def ip_lookup(ip):
        result = search_tree[ip]
        return result.pop().data if result else 'Others'

    data_fraud['country'] = data_fraud['ip_address'].apply(ip_lookup)

    return data_fraud


# create a new coloumn because some fraudulent activities can have short duration between signup and purchase
def time_differences(data_fraud):
    data_fraud['time_difference'] = (data_fraud['purchase_time'] - data_fraud['signup_time'])
    data_fraud['time_difference_sec'] = data_fraud['time_difference'].dt.total_seconds().astype(int)
    return data_fraud


# purchase time
def purchase_time(data_fraud):
    data_fraud['day_purchase'] = data_fraud['purchase_time'].dt.day
    data_fraud['hour_purchase'] = data_fraud['purchase_time'].dt.hour
    data_fraud['minute_purchase'] = data_fraud['purchase_time'].dt.minute
    data_fraud['second_purchase'] = data_fraud['purchase_time'].dt.second
    return data_fraud


def device_id_frequency(data_fraud):
    data_fraud['device_id_frequency'] = data_fraud.groupby('device_id')['device_id'].transform('count')
    return data_fraud
def country_frequency(data_fraud):
    data_fraud['country_frequency'] = data_fraud.groupby('country')['country'].transform('count')
    return data_fraud

def drop_clean_data(data_fraud):
    columns_drop = ['signup_time', 'purchase_time', 'device_id', 'time_difference', 'ip_address', 'country']
    data_fraud = data_fraud.drop(columns_drop, axis=1)
    dependent_variable = 'class'
    data_fraud = data_fraud[[col for col in data_fraud.columns if col != dependent_variable] + [dependent_variable]]
    return data_fraud


if __name__ == '__main__':
    data_url = 'C:\\Users\\icicala\\Desktop\\Thesis\\Thesis\\Data\\Fraud_Data.csv'
    ip_url = 'C:\\Users\\icicala\\Desktop\\Thesis\\Thesis\\Data\\IpAddress_to_Country.csv'
    final_url = 'C:\\Users\\icicala\\Desktop\\Thesis\\Thesis\\Data\\EFraud_data.csv'

    # count_minus_1 = (data_ip['lower_bound_ip_address'] == -1).sum()
    # print(f"Number of rows with '-1' in 'lower_bound_ip_address' column: {count_minus_1}")

    # rows_with_minus_1 = data_ip[data_ip['lower_bound_ip_address'] == -1]
    # print(rows_with_minus_1)

    # data_fraud = load_data(data_url)
    # print(data_fraud.info())

    data_fraud = load_data(data_url)
    data_ip = load_ip(ip_url)
    reformat_data = replace_Ip_country(data_fraud, data_ip)
    time_data = time_differences(reformat_data)
    purchase_time = purchase_time(time_data)
    device_id_data = device_id_frequency(purchase_time)
    country_data = country_frequency(device_id_data)
    final_data = drop_clean_data(device_id_data)
    final_data.to_csv(final_url, index=False)


    pd.set_option('display.max_columns', None)
    print(final_data)
