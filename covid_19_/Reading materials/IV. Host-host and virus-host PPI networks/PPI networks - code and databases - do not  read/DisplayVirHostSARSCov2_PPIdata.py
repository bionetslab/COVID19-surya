import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

G = nx.read_graphml('IID.graphml')
nx.draw(G)
plt.show()
########################################################
for edge in G.edges(data=True):
    print(edge)



for node in G.nodes(data=True):
    print(node)
##########################################
