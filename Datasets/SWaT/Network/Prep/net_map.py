import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

filename = './net_map.csv'
df = pd.read_csv(filename, header=0)
df[['src', 'dst']].drop_duplicates()
df.dropna()
G = nx.from_pandas_edgelist(df, 'src', 'dst', create_using=nx.DiGraph())
plt.figure(3,figsize=(60,60))
nx.draw(G, with_labels=True, node_size=7000, pos=nx.kamada_kawai_layout(G), arrowsize=20)

plt.show()