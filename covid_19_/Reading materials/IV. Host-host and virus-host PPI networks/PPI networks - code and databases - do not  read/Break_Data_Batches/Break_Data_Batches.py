import pandas as pd

source_path = "andas.txt"

for i,chunk in enumerate(pd.read_csv(source_path, chunksize=10000)):
    chunk.to_csv('chunk{}.txt'.format(i), index=False, header=None)