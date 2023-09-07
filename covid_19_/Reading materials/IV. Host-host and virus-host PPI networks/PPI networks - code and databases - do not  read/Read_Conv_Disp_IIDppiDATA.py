import networkx as nx
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import csv
import numpy as np
from pandas import DataFrame

fields = ['symbol1', 'symbol2']
data = pd.read_csv('PPI_IID_FINAL.txt', usecols=fields, sep="\t")


# convert to dataframe using the first row as the column names; drop empty, final row
df = pd.DataFrame(data[1:-1], columns = ['symbol1','symbol2']) 
# # dataframe with the preferred names of the two proteins and the score of the interaction
interactions = df[['symbol1','symbol2']]

G=nx.Graph(name='Protein Interaction Graph')
interactions = np.array(interactions)
for i in range(len(interactions)):
    interaction = interactions[i]
    a = interaction[0] # protein a node
    b = interaction[1] # protein b node
    #w = float(interaction[2]) # score as weighted edge where high scores = low weight
    #G.add_weighted_edges_from([(a,b)]) # add weighted edge to graph
    G.add_edges_from([(a,b)])

#############################################################################################

# nx.draw(G)
# plt.show()

# ##########################################################################################

# for edge in G.edges(data=True):
#     print(edge)



# for node in G.nodes(data=True):
#     print(node)

#########################################

# # Here is some basic information about the graph using nx.info(G):
# print(nx.info(G))

###############################