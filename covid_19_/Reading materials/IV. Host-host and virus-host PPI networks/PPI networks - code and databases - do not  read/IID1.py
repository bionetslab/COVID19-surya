# import pandas as pd
# X = pd.read_csv('test.txt', sep="\t", header=None)


import csv
import numpy as np
import pandas as pd
from pandas import DataFrame

fields = ['uniprot1', 'uniprot2']

csv_reader = pd.read_csv('human_annotated_PPIs.txt', usecols=fields, sep="\t")
# csv_reader_COL1=csv_reader['uniprot1']
# print(csv_reader_COL1.size)


# csv_reader.to_csv(r'andas.txt', header=None, index=None, sep='\t', mode='a')
csv_reader.to_csv(r'andas.txt', index=None, sep='\t', mode='a')