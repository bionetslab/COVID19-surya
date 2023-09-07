import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_graphml('HEK293T_SARS-CoV-2.graphml')
nx.draw(G)
plt.show()