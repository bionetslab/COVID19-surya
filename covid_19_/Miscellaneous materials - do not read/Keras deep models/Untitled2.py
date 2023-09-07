import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import imageio
import pandas as pd
import random
import pickle
import pandas as pd

random.seed(100)

folder='/Volumes/Seagate Exp/CellImagingMainProject/'


# ############################################################################
# ############################################################################

# # strng=folder+'X_SevereCytokine2700samples_64x64_3d.npy'
# strng=folder+'X_SevereCytokine2700_224x224_3d_RGB.npy'
# X_SevereCytokine2700=np.load(strng)
# len_X_SevereCytokine2700=len(X_SevereCytokine2700)
# random.shuffle(X_SevereCytokine2700)

# # strng=folder+'X_Healthy_64x64_3d.npy'
# strng=folder+'X_Healthy_224x224_3d_RGB.npy'
# X_Healthy=np.load(strng)
# len_X_Healthy=len(X_Healthy)
# random.shuffle(X_Healthy)

# X=[]
# y=[]
# for i in range(len_X_SevereCytokine2700):
#     X.append(X_SevereCytokine2700[i])
#     y.append(int(1))
# for i in range(len_X_Healthy):
#     X.append(X_Healthy[i])
#     y.append(int(0))
    
# def Shuffle(X, y):
#     X_shuffled=[]
#     y_shuffled=[]
#     length=len(y)
#     index=list(range(length))
#     random.Random(12).shuffle(index)
#     for i in range(length):
#         X_shuffled.append(X[index[i]])
#         y_shuffled.append(y[index[i]])
#     return X_shuffled, y_shuffled
    
# X, y=Shuffle(X, y)
# # str=folder + 'X_SevereCytokine2700_Healthy_randomizedFinal_64x64.npy'
# str=folder + 'X_SevereCytokine2700_Healthy_randomizedFinal_224x224_RGB.npy'
# np.save(str, X, allow_pickle=True)
# # str=folder + 'y_SevereCytokine2700_Healthy_randomizedFinal_64x64.npy'
# str=folder + 'y_SevereCytokine2700_Healthy_randomizedFinal_224x224_RGB.npy'
# np.save(str, y, allow_pickle=True)

# ############################################################################
# ############################################################################

# strng=folder+'X_SevereCytokine260samples_64x64_3d.npy'
strng=folder+'X_SevereCytokine260_224x224_3d_RGB.npy'
X_SevereCytokine260=np.load(strng)
len_X_SevereCytokine260=len(X_SevereCytokine260)
random.shuffle(X_SevereCytokine260)

# strng=folder+'X_NoCytokine_64x64_3d.npy'
strng=folder+'X_NoCytokine_224x224_3d_RGB.npy'
X_NoCytokine=np.load(strng)
len_X_NoCytokine=len(X_NoCytokine)
random.shuffle(X_NoCytokine)

X=[]
y=[]
for i in range(len_X_SevereCytokine260):
    X.append(X_SevereCytokine260[i])
    y.append(int(1))
for i in range(len_X_NoCytokine):
    X.append(X_NoCytokine[i])
    y.append(int(0))
    
def Shuffle(X, y):
    X_shuffled=[]
    y_shuffled=[]
    length=len(y)
    index=list(range(length))
    random.Random(12).shuffle(index)
    for i in range(length):
        X_shuffled.append(X[index[i]])
        y_shuffled.append(y[index[i]])
    return X_shuffled, y_shuffled
    
X, y=Shuffle(X, y)
# # str=folder + 'X_SevereCytokine260_NoCytokine_randomizedFinal_64x64.npy'
str=folder + 'X_SevereCytokine260_NoCytokine_randomizedFinal_224x224_RGB.npy'
np.save(str, X, allow_pickle=True)
# # str=folder + 'y_SevereCytokine260_NoCytokine_randomizedFinal_64x64.npy'
str=folder + 'y_SevereCytokine260_NoCytokine_randomizedFinal_224x224_RGB.npy'
np.save(str, y, allow_pickle=True)