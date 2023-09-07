import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

G = nx.read_graphml('IID.graphml')
nx.draw(G)
plt.show()

#########################################################################
#########################################################################

nodes_G=G.nodes()
print(nodes_G)
print(type(nodes_G))

#########################################################################
#########################################################################

H = nx.read_graphml('HEK293T_SARS-CoV-2.graphml')
nx.draw(H)
plt.show()

count=0
for edge in H.edges(data=True):
  count=count+1
  print(edge)
print(count)


nodes = H.nodes(data=True)
count=0
for src, tgt, attr in H.edges(data=True):
  count=count+1
  # print type(node1)
  src_attr = nodes[src]
  tgt_attr = nodes[tgt]
  ###### print(str(src) + ' - ' + str(tgt) + ' : ' + attr['name'] + ' : ' + src_attr['name'] + ' , ' + tgt_attr['name'])
  # print(src_attr['name'] + ' , ' + tgt_attr['name'])
  textDATA=src_attr['name'] + '\t' + tgt_attr['name']
  print(textDATA)
print(count)
# Here is some basic information about the graph using nx.info(G):
print(nx.info(H))
#######################################################################################################################################
import networkx as nx
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import csv
import numpy as np
from pandas import DataFrame

fields = ['name1', 'name2']
data = pd.read_csv('SARS-Cov2-VirHostPPIData.txt', usecols=fields, sep="\t")


# convert to dataframe using the first row as the column names; drop empty, final row
df = pd.DataFrame(data[1:-1], columns = ['name1','name2']) 
# # dataframe with the preferred names of the two proteins and the score of the interaction
interactions = df[['name1','name2']]

I=nx.Graph(name='SARS Cov2 VirHostPPIData')
interactions = np.array(interactions)
for i in range(len(interactions)):
    interaction = interactions[i]
    a = interaction[0] # protein a node
    b = interaction[1] # protein b node
    #w = float(interaction[2]) # score as weighted edge where high scores = low weight
    #G.add_weighted_edges_from([(a,b)]) # add weighted edge to graph
    I.add_edges_from([(a,b)])

nx.draw(I)
plt.show()

# ##########################################################################################

for edge in I.edges(data=True):
  print(edge)



for node in I.nodes(data=True):
  print(node)

#########################################

# Here is some basic information about the graph using nx.info(G):
print(nx.info(I))


####################################################################################################
####################################################################################################

nodes_I=I.nodes()
print(nodes_I)
print(type(nodes_I))

####################################################################################################
####################################################################################################

edges_I=I.edges()
print(edges_I)
print(type(edges_I))

####################################################################################################
####################################################################################################

edges_I_List=list(edges_I)
print(edges_I_List)
print(type(edges_I_List))

####################################################################################################
####################################################################################################

print(np.shape(edges_I_List))

####################################################################################################
####################################################################################################

nodes_I_=list(nodes_I)
print(nodes_I_)
print(type(nodes_I_))

nodes_G_=list(nodes_G)
print(nodes_G_)
print(type(nodes_G_))

####################################################################################################
####################################################################################################

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

print(intersection(nodes_I_, nodes_G_))
print(len(intersection(nodes_I_, nodes_G_)))

####################################################################################################
####################################################################################################

J=nx.compose(G,I)

########################################################
########################################################

nx.draw(J)
plt.show()

########################################################
########################################################

nodes_J=J.nodes()
print(nodes_J)
print(type(nodes_J))
print('##################')
nodes_J_=list(nodes_J)
print(nodes_J_)
print(type(nodes_J_))
print('##################')
print(np.shape(nodes_J_))

########################################################
########################################################

# for node in J.nodes(data=True):
    # print(node('name'))
# nx.get_node_attributes(J,'name')

nodes = J.nodes(data=True)
count=0
NODES_src=[]
NODES_tgt=[]
for src, tgt, attr in J.edges(data=True):
  count=count+1
  NODES_src.append(src)
  NODES_tgt.append(tgt)
  # print type(node1)
  src_attr = nodes[src]
  tgt_attr = nodes[tgt]
  ###### print(str(src) + ' - ' + str(tgt) + ' : ' + attr['name'] + ' : ' + src_attr['name'] + ' , ' + tgt_attr['name'])
  # print(src)
  # textDATA=src_attr['name'] + '\t' + tgt_attr['name']
  # print(textDATA)
print(count)
print(src)
print(type(src))
print(np.shape(src))
print(len(src))
print(np.shape(NODES_src))
print('##########')
print(tgt)
print(type(tgt))
print(np.shape(tgt))
print(len(tgt))
print(NODES_tgt)
print(type(NODES_tgt))

########################################################
########################################################

print(src_attr)

########################################################
########################################################

noOfNodes=count
# for node in J.nodes(data=True):
    # print(node('name'))
# nx.get_node_attributes(J,'name')
EDGE_INDEXING=np.zeros((count,2))
EDGE_INDEXING=EDGE_INDEXING.astype(int)
nodes = J.nodes(data=True)
count=0
for src, tgt, attr in J.edges(data=True):
  
  # print type(node1)
  src_attr = nodes[src]
  tgt_attr = nodes[tgt]
  
  
  EDGE_INDEXING[count,0]=nodes_J_.index(src)
  EDGE_INDEXING[count,1]=nodes_J_.index(tgt)
  
  
  
  # print(src + ' - ' + tgt)
  # print(src_attr['name'] + ' , ' + tgt_attr['name'])
  # textDATA=src_attr['name'] + '\t' + tgt_attr['name']
  # print(textDATA)
  count=count+1
# print(count)


########################################################
########################################################

print(EDGE_INDEXING)
#######################
print(type(EDGE_INDEXING))
#######################
print(np.shape(EDGE_INDEXING))

#################################################
#################################################

# ! pip install pcst_fast

##################################################
##################################################

Shape=np.shape(EDGE_INDEXING)
print(Shape)
print(type(Shape))

##################################################
##################################################

noOfEdges=Shape[0]

import random

costs=np.zeros((noOfEdges))


for i in range (noOfEdges):
  costs[i]=random.uniform(0, 1)

##################################################
##################################################

prizes=np.zeros((noOfNodes))


for i in range (noOfNodes):
  prizes[i]=random.uniform(0, 1)

##################################################
##################################################

print(np.shape(prizes))

##################################################
##################################################

edges=EDGE_INDEXING

import pcst_fast
op_vertices, op_edges = pcst_fast.pcst_fast(edges, prizes, costs, -1, 1, 'strong', 0)

##################################################
##################################################

print(edges)

##################################################
##################################################

print(np.unique(prizes))

##################################################
##################################################

print(costs)

##################################################
##################################################

print(op_vertices)

##################################################
##################################################

print(op_edges)

##################################################
##################################################

print(type(src))

##################################################
##################################################


































































































































































































