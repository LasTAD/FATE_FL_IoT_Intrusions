from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

training_df = pd.read_csv('KDDTrain+.csv', header=None)
testing_df = pd.read_csv('KDDTest+.csv', header=None)

columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome',
    'difficulty'
]
training_df.columns = columns
testing_df.columns = columns

# A list ot attack names that belong to each general attack type
dos_attacks = ["snmpgetattack", "back", "land", "neptune", "smurf", "teardrop", "pod", "apache2", "udpstorm",
               "processtable", "mailbomb"]
r2l_attacks = ["snmpguess", "worm", "httptunnel", "named", "xlock", "xsnoop", "sendmail", "ftp_write", "guess_passwd",
               "imap", "multihop", "phf", "spy", "warezclient", "warezmaster"]
u2r_attacks = ["sqlattack", "buffer_overflow", "loadmodule", "perl", "rootkit", "xterm", "ps"]
probe_attacks = ["ipsweep", "nmap", "portsweep", "satan", "saint", "mscan"]

# Our new labels
classes = ["Normal", "Dos", "R2L", "U2R", "Probe"]


# Helper function to label samples to 5 classes
def label_attack(row):
    if row["outcome"] in dos_attacks:
        return classes[1]
    if row["outcome"] in r2l_attacks:
        return classes[2]
    if row["outcome"] in u2r_attacks:
        return classes[3]
    if row["outcome"] in probe_attacks:
        return classes[4]
    return classes[0]


# We combine the datasets temporarily to do the labeling
test_samples_length = len(testing_df)
df = pd.concat([training_df, testing_df])
df["Class"] = df.apply(label_attack, axis=1)

# The old outcome field is dropped since it was replaced with the Class field,
# the difficulty field will be dropped as well.
df = df.drop("outcome", axis=1)
df = df.drop("difficulty", axis=1)

# we again split the data into training and test sets.
training_df = df.iloc[:-test_samples_length, :]
testing_df = df.iloc[-test_samples_length:, :]


# Helper function for scaling continous values
def minmax_scale_values(training_df, testing_df, col_name):
    scaler = MinMaxScaler()
    scaler = scaler.fit(training_df[col_name].values.reshape(-1, 1))
    train_values_standardized = scaler.transform(training_df[col_name].values.reshape(-1, 1))
    training_df[col_name] = train_values_standardized
    test_values_standardized = scaler.transform(testing_df[col_name].values.reshape(-1, 1))
    testing_df[col_name] = test_values_standardized


# Helper function for one hot encoding
def encode_text(training_df, testing_df, name):
    training_set_dummies = pd.get_dummies(training_df[name])
    testing_set_dummies = pd.get_dummies(testing_df[name])
    for x in training_set_dummies.columns:
        dummy_name = "{}_{}".format(name, x)
        training_df[dummy_name] = training_set_dummies[x]
        if x in testing_set_dummies.columns:
            testing_df[dummy_name] = testing_set_dummies[x]
        else:
            testing_df[dummy_name] = np.zeros(len(testing_df))
    training_df.drop(name, axis=1, inplace=True)
    testing_df.drop(name, axis=1, inplace=True)


sympolic_columns = ["protocol_type", "service", "flag"]
label_column = "Class"
for column in df.columns:
    if column in sympolic_columns:
        encode_text(training_df, testing_df, column)
    elif not column == label_column:
        minmax_scale_values(training_df, testing_df, column)

print(training_df.head(5))
print(len(training_df))
print(testing_df.head(5))
print(len(testing_df))
training_df_federated = np.array_split(training_df, 4)
i = 1
for part in training_df_federated:
    part.to_csv('nsl_kdd_train_' + str(i) + '.csv')
    print(len(part))
    i += 1
print('\n')
testing_df_federated = np.array_split(training_df, 4)
i = 1
for part in training_df_federated:
    part.to_csv('nsl_kdd_test_' + str(i) + '.csv')
    print(len(part))
    i += 1
