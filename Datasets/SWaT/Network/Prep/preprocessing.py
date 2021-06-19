import glob
import os

import pandas as pd

path = '../raw_data'
all_files = glob.glob(os.path.join(path, "*.csv"))
li = []
i = 1
allEl = len(all_files)
for filename in all_files:
    print('reading of ' + filename)
    df = pd.read_csv(filename, index_col='num', header=0, low_memory=False, error_bad_lines=False)
    # df = df[pd.notnull(df['SCADA_Tag'])]
    li.append(df[['SCADA_Tag', 'src', 'dst']].drop_duplicates())
    print('remaining ' + str(allEl - i) + ' files')
    i += 1
frame = pd.concat(li, axis=0, ignore_index=True)
frame[['SCADA_Tag', 'src', 'dst']].drop_duplicates()
frame.to_csv('net_map.csv', index=False)
