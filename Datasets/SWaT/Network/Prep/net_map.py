import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

filename = './net_map.csv'
df = pd.read_csv(filename, header=0)
df['num'] = np.log10(df['num'])
most_net = df.loc[df['num'] >= 3]
G = nx.from_pandas_edgelist(most_net, 'src', 'dst', edge_attr=['num'], create_using=nx.MultiDiGraph())
fig, ax = plt.subplots()
fig.set_figheight(10)
fig.set_figwidth(10)
ax.set_title('Карта сети')
nx.draw_networkx(G, ax=ax, with_labels=True, node_size=7000, width=most_net['num'].values.astype(float),
                 pos=nx.kamada_kawai_layout(G), arrowsize=20)

plt.savefig('../maps/net_map_kawai_more1000.png')
