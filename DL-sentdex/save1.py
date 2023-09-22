import numpy as np
from numpy import save
n1=np.load('X_rxrx1_images_0to5000.npy')
n2=np.load('X_rxrx1_images_5000to10000.npy')
N=np.concatenate((n1, n2))
save('X_rxrx1_images_0to10000_new.npy', N)