import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

ds = pd.read_csv('Physical/prep/physical_data_raw.csv', header=0)
print('Read complete\n')

ds = ds.drop('Timestamp', axis=1)
print(ds.head(5))
train, test = train_test_split(ds, test_size=0.2, stratify=ds['Normal/Attack'])
print('Splitting complete\n')


def minmax_scale_values(training_df, testing_df, col_name):
    scaler = MinMaxScaler()
    scaler = scaler.fit(training_df[col_name].values.reshape(-1, 1))
    train_values_standardized = scaler.transform(training_df[col_name].values.reshape(-1, 1))
    training_df[col_name] = train_values_standardized
    test_values_standardized = scaler.transform(testing_df[col_name].values.reshape(-1, 1))
    testing_df[col_name] = test_values_standardized


label_column = "Normal/Attack"
for column in ds.columns:
    if column != label_column:
        minmax_scale_values(train, test, column)
print('Scaling complete\n')

# train.to_csv('Physical/to_FATE/hetero/swat_phys_train_p1.csv', columns=['FIT101', 'LIT101', "MV101", "P101", "P102"])
# train.to_csv('Physical/to_FATE/hetero/swat_phys_train_p2.csv',
#              columns=['AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206'])
# train.to_csv('Physical/to_FATE/hetero/swat_phys_train_p3.csv', columns=['DPIT301',
#                                                                         'FIT301',
#                                                                         'LIT301',
#                                                                         'MV301',
#                                                                         'MV302',
#                                                                         'MV303',
#                                                                         'MV304',
#                                                                         'P301',
#                                                                         'P302'])
# train.to_csv('Physical/to_FATE/hetero/swat_phys_train_p4.csv',
#              columns=['AIT401',
#                       'AIT402',
#                       'FIT401',
#                       'LIT401',
#                       'P401',
#                       'P402',
#                       'P403',
#                       'P404',
#                       'UV401'])
# train.to_csv('Physical/to_FATE/hetero/swat_phys_train_p5.csv', columns=['AIT501',
#                                                                         'AIT502',
#                                                                         'AIT503',
#                                                                         'AIT504',
#                                                                         'FIT501',
#                                                                         'FIT502',
#                                                                         'FIT503',
#                                                                         'FIT504',
#                                                                         'P501',
#                                                                         'P502',
#                                                                         'PIT501',
#                                                                         'PIT502',
#                                                                         'PIT503'])
# train.to_csv('Physical/to_FATE/hetero/swat_phys_train_p6_label.csv', columns=["FIT601",
#                                                                               'P601',
#                                                                               'P602',
#                                                                               'P603',
#                                                                               'Normal/Attack'])
# train.to_csv('Physical/to_FATE/hetero/swat_phys_train_label.csv', columns=['Normal/Attack'])
# print('Train datasets ready')
# print(len(train))
#
# test.to_csv('Physical/to_FATE/hetero/swat_phys_test_p1.csv', columns=['FIT101', 'LIT101', 'MV101', 'P101', 'P102'])
# test.to_csv('Physical/to_FATE/hetero/swat_phys_test_p2.csv',
#             columns=['AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206'])
# test.to_csv('Physical/to_FATE/hetero/swat_phys_test_p3.csv', columns=['DPIT301',
#                                                                       'FIT301',
#                                                                       'LIT301',
#                                                                       'MV301',
#                                                                       'MV302',
#                                                                       'MV303',
#                                                                       'MV304',
#                                                                       'P301',
#                                                                       'P302'])
# test.to_csv('Physical/to_FATE/hetero/swat_phys_test_p4.csv',
#             columns=['AIT401',
#                      'AIT402',
#                      'FIT401',
#                      'LIT401',
#                      'P401',
#                      'P402',
#                      'P403',
#                      'P404',
#                      'UV401'])
# test.to_csv('Physical/to_FATE/hetero/swat_phys_test_p5.csv', columns=['AIT501',
#                                                                       'AIT502',
#                                                                       'AIT503',
#                                                                       'AIT504',
#                                                                       'FIT501',
#                                                                       'FIT502',
#                                                                       'FIT503',
#                                                                       'FIT504',
#                                                                       'P501',
#                                                                       'P502',
#                                                                       'PIT501',
#                                                                       'PIT502',
#                                                                       'PIT503'])
# test.to_csv('Physical/to_FATE/hetero/swat_phys_test_p6_label.csv', columns=["FIT601",
#                                                                             'P601',
#                                                                             'P602',
#                                                                             'P603',
#                                                                             'Normal/Attack'])
# test.to_csv('Physical/to_FATE/hetero/swat_phys_test_label.csv', columns=['Normal/Attack'])
# print('Test datasets ready')
# print(len(test))

pd.concat([train, test], axis=0)
train.to_csv('Physical/to_FATE/full/swat_phys.csv', index=True, index_label='id')
