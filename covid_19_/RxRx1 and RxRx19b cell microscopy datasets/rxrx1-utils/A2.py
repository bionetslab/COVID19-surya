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
from pathlib import Path
import sys



# ###########################################################


# folder='/Volumes/Seagate Exp/CellImagingMainProject/'

# strng=folder+'X_SevereCytokine260samples_512x512_3d.npy'
# X_SevereCytokine260=np.load(strng)

# # ------------------------------------------------------------

# sys.path.append('/Users/surya/Desktop/CellImagingRxRx/rxrx1-utils/rxrx1-utils-main')
# import rxrx.io as rio

# X_SevereCytokine260_3d_RGB=[]
# for i in range(len(X_SevereCytokine260)):
#     X_SevereCytokine260_3d_RGB.append(rio.convert_tensor_to_rgb(X_SevereCytokine260[i]))

# target_folder='/Volumes/Seagate Exp/CellImagingMainProject/'
# str=target_folder + 'X_SevereCytokine260_512x512_3d_RGB.npy'
# np.save(str, X_SevereCytokine260_3d_RGB, allow_pickle=True)




# ###########################################################


# folder='/Volumes/Seagate Exp/CellImagingMainProject/'

# strng=folder+'X_SevereCytokine2700samples_512x512_3d.npy'
# X_SevereCytokine2700=np.load(strng)

# # ------------------------------------------------------------

# sys.path.append('/Users/surya/Desktop/CellImagingRxRx/rxrx1-utils/rxrx1-utils-main')
# import rxrx.io as rio

# X_SevereCytokine2700_3d_RGB=[]
# for i in range(len(X_SevereCytokine2700)):
#     X_SevereCytokine2700_3d_RGB.append(rio.convert_tensor_to_rgb(X_SevereCytokine2700[i]))

# target_folder='/Volumes/Seagate Exp/CellImagingMainProject/'
# str=target_folder + 'X_SevereCytokine2700_512x512_3d_RGB.npy'
# np.save(str, X_SevereCytokine2700_3d_RGB, allow_pickle=True)




# ###########################################################


# folder='/Volumes/Seagate Exp/CellImagingMainProject/'

# strng=folder+'X_NoCytokine_512x512_3d.npy'
# X_NoCytokine=np.load(strng)

# # ------------------------------------------------------------

# sys.path.append('/Users/surya/Desktop/CellImagingRxRx/rxrx1-utils/rxrx1-utils-main')
# import rxrx.io as rio

# X_NoCytokine_3d_RGB=[]
# for i in range(len(X_NoCytokine)):
#     X_NoCytokine_3d_RGB.append(rio.convert_tensor_to_rgb(X_NoCytokine[i]))

# target_folder='/Volumes/Seagate Exp/CellImagingMainProject/'
# str=target_folder + 'X_NoCytokine_512x512_3d_RGB.npy'
# np.save(str, X_NoCytokine_3d_RGB, allow_pickle=True)





# ###########################################################


# folder='/Volumes/Seagate Exp/CellImagingMainProject/'

# strng=folder+'X_Healthy_512x512_3d.npy'
# X_Healthy=np.load(strng)

# # ------------------------------------------------------------

# sys.path.append('/Users/surya/Desktop/CellImagingRxRx/rxrx1-utils/rxrx1-utils-main')
# import rxrx.io as rio

# X_Healthy_3d_RGB=[]
# for i in range(len(X_Healthy)):
#     X_Healthy_3d_RGB.append(rio.convert_tensor_to_rgb(X_Healthy[i]))

# target_folder='/Volumes/Seagate Exp/CellImagingMainProject/'
# str=target_folder + 'X_Healthy_512x512_3d_RGB.npy'
# np.save(str, X_Healthy_3d_RGB, allow_pickle=True)








