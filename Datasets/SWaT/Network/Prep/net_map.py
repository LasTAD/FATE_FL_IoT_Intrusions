import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

filename = './net_map.csv'
df = pd.read_csv(filename, header=0)
G = nx.from_pandas_edgelist(df, 'src', 'dst', edge_attr=['num'], create_using=nx.MultiDiGraph())
plt.figure(3, figsize=(60, 60))
nx.draw_networkx(G, with_labels=True, node_size=7000, pos=nx.kamada_kawai_layout(G), arrowsize=20)
plt.savefig('../maps/net_map_kawai.png')
nx.draw(G, with_labels=True, node_size=7000, pos=nx.spring_layout(G), arrowsize=20)
plt.savefig('../maps/net_map_spring.png')
nx.draw(G, with_labels=True, node_size=7000, pos=nx.fruchterman_reingold_layout(G), arrowsize=20)
plt.savefig('../maps/net_map_reingold.png')
