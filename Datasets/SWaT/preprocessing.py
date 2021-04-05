import pandas as pd
import os
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://localhost/swat")

ds = pd.read_csv('Physical/SWaT_Dataset_Attack_v0.csv', parse_dates=[0], header=0, sep=';', decimal=",")
print('Read complete')
ds.to_sql('physical_data_raw', engine, index=False, if_exists='append')
print(ds.head(5))

# ds = pd.read_csv('Physical/SWaT_Dataset_Normal_v0.csv', parse_dates=[0], header=0, sep=';', decimal=",")
# list_of_files = os.listdir("Network")
# num = len(list_of_files)
# print(num)
# for file in list_of_files:
#     ds = pd.read_csv('Network/' + file, parse_dates=[[1, 2]], header=0, sep=',', decimal=",", index_col='num')
#     print(file + ' readed')
#     ds.to_sql('network_data_raw', engine, index=False, if_exists='append')
#     print(file + ' committed')
#     num = num - 1
#     print(num)
