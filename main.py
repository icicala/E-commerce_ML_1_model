# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# Identifying missing values in tabular data
def missing_values():
    dataframe = load_data()
    print(dataframe.isnull().sum())


# 1.1 Basic information of data set - data distribution and find outliers
def check_data():
    dataframe = load_data()
    pd.set_option('display.max_columns', None)
    print('#################################')
    print(dataframe)
    print('#################################')
    print(dataframe.describe())
    print('#################################')
    print(dataframe.head())
    print('#################################')
    print(dataframe.info())


# Sequential feature selection - recursive backward elimination
# TO DO

# Univariate Analysis -label
def label_analysis():
    dataframe = load_data()
    pd.set_option('display.max_columns', None)
    print(dataframe['class'].describe())
    class_percentage = dataframe['class'].value_counts(normalize=True) * 100
    plt.figure(figsize=(9, 8))
    ax = sns.barplot(x=class_percentage.index, y=class_percentage.values, color='black', alpha=0.8)
    ax.set_title("Distribution of Class")
    ax.set_xlabel('Class')
    ax.set_ylabel("Frequency")
    plt.show()


# Univariate Analysis - continuous
def continous_analysis():
    dataframe = load_data()
    dataframe['purchase_value'].info()
    dataframe['age'].info()
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64'])
    numeric_columns = numeric_columns[numeric_columns.columns.tolist()[1:3]]
    numeric_columns.hist(figsize=(10, 10), bins=100, xlabelsize=18, ylabelsize=18)
    plt.show()


# Univariate Analysis - category
def category_analysis():
    dataframe = load_data()
    category_columns = dataframe.select_dtypes(include=['object'])
    category_columns = category_columns[category_columns.columns.tolist()[3:]]
    print(category_columns.columns)
    for column in category_columns.columns:
        # column = '' # source, browser, sex
        plt.figure(figsize=(8, 6))
        dataframe[column].value_counts().plot(kind='bar', color='skyblue')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()


# category devide_id
def category_device_id_analysis():
    dataframe = load_data()
    device_id = dataframe['device_id'].value_counts()
    count_summary = device_id.value_counts()
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=count_summary.index, y=count_summary.values, color='skyblue')
    plt.title('Histogram of Device_ID Counts')
    plt.xlabel('Count of Device_ID')
    plt.ylabel('Frequency')
    # Add count numbers on top of each bar
    for i, count in enumerate(count_summary.values):
        ax.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=10)

    plt.show()


# return a country
# def country_from_IP(ip, ip_dataframe):
#     try:
#         ip_int = int(ip)
#         country_condition = (ip_int <= ip_dataframe['upper_bound_ip_address']) & (
#                     ip_int >= ip_dataframe['lower_bound_ip_address'])
#         matching_countries = ip_dataframe.loc[country_condition, 'country']
#
#         if not matching_countries.empty:
#             country = matching_countries.iloc[0]
#             return country
#         else:
#             return "Unknown"
#     except ValueError:
#         return "Invalid IP"

def country_from_IP(ip, ip_dataframe):
    try:
        ip_int = int(ip)
        country_condition = (ip_int <= ip_dataframe['upper_bound_ip_address']) & (
                ip_int >= ip_dataframe['lower_bound_ip_address'])
        matching_index = np.argmax(country_condition.to_numpy())
        if matching_index:
            country = ip_dataframe.loc[matching_index, 'country']
            return country
        else:
            return "Unknown"
    except ValueError:
        return "Invalid IP"

# need to fix the errors
def ip_address_categorical_analysis():

    ip_dataframe = pd.read_csv('C:\\Users\\icicala\\Desktop\\Thesis\\Thesis\Data\\IpAddress_to_Country.csv')
    ip_dataframe = ip_dataframe.sort_values('lower_bound_ip_address')
    dataframe = load_data()
    dataframe['country'] = dataframe['ip_address'].apply(lambda i: country_from_IP(i, ip_dataframe))
    coutry_counts = dataframe['country'].value_counts()
    coutry_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Users by Country')
    plt.xlabel('Country')
    plt.ylabel('Number of Users')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



# Analyze the relationship/correlation between variables and labels
def corr_analisys():
    dataframe = load_data()


def load_data():
    dataframe = pd.read_csv('C:\\Users\\icicala\\Desktop\\Thesis\\Thesis\Data\\Fraud_Data.csv')
    return dataframe


if __name__ == '__main__':
    start_time = time.time()
    ip_address_categorical_analysis()
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")
