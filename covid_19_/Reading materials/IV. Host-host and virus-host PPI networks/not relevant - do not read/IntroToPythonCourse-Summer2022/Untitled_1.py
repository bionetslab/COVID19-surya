import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
from PIL import Image
import imageio
import pandas as pd
import random
import pickle
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

random.seed(100)
IMG_SIZE=224

# #######################################################################################
# #######################################################################################

# target_folder='/Volumes/Seagate Exp/CellImagingMainProject/'
# str=target_folder + 'X_SevereCytokine260_512x512_3d_RGB.npy'
# X_SevereCytokine260_512x512_3d_RGB=np.load(str, allow_pickle=True)


# X_SevereCytokine260_224x224_3d_RGB=[]


# for i in range(len(X_SevereCytokine260_512x512_3d_RGB)):
#     basewidth = IMG_SIZE
#     img1=Image.fromarray((X_SevereCytokine260_512x512_3d_RGB[i,:,:,0]).astype(np.uint8))
#     img2=Image.fromarray((X_SevereCytokine260_512x512_3d_RGB[i,:,:,1]).astype(np.uint8))
#     img3=Image.fromarray((X_SevereCytokine260_512x512_3d_RGB[i,:,:,2]).astype(np.uint8))
#     # -------------------------
#     wpercent = float(basewidth / 512)
#     hsize = int(512 * wpercent)
#     img1 = img1.resize((basewidth, hsize), Image.ANTIALIAS)
#     img2 = img2.resize((basewidth, hsize), Image.ANTIALIAS)
#     img3 = img3.resize((basewidth, hsize), Image.ANTIALIAS)
#     # -------------------------
#     img1 = np.array(img1)
#     img2 = np.array(img2)
#     img3 = np.array(img3)
#     # -------------------------
#     img_array=cv2.merge([img1, img2, img3])
#     # -------------------------
#     X_SevereCytokine260_224x224_3d_RGB.append(img_array)
# random.shuffle(X_SevereCytokine260_224x224_3d_RGB)
# x_SevereCytokine260_224x224_3d_RGB=[]
# for features in X_SevereCytokine260_224x224_3d_RGB:
#     x_SevereCytokine260_224x224_3d_RGB.append(features)
# x_SevereCytokine260_224x224_3d_RGB=np.array(x_SevereCytokine260_224x224_3d_RGB).reshape(-1,IMG_SIZE,IMG_SIZE,3)
# target_folder='/Volumes/Seagate Exp/CellImagingMainProject/'
# str=target_folder + 'X_SevereCytokine260_224x224_3d_RGB.npy'
# np.save(str, x_SevereCytokine260_224x224_3d_RGB, allow_pickle=True)

# #######################################################################################
# #######################################################################################

# target_folder='/Volumes/Seagate Exp/CellImagingMainProject/'
# str=target_folder + 'X_SevereCytokine2700_512x512_3d_RGB.npy'
# X_SevereCytokine2700_512x512_3d_RGB=np.load(str, allow_pickle=True)


# X_SevereCytokine2700_224x224_3d_RGB=[]


# for i in range(len(X_SevereCytokine2700_512x512_3d_RGB)):
#     basewidth = IMG_SIZE
#     img1=Image.fromarray((X_SevereCytokine2700_512x512_3d_RGB[i,:,:,0]).astype(np.uint8))
#     img2=Image.fromarray((X_SevereCytokine2700_512x512_3d_RGB[i,:,:,1]).astype(np.uint8))
#     img3=Image.fromarray((X_SevereCytokine2700_512x512_3d_RGB[i,:,:,2]).astype(np.uint8))
#     # -------------------------
#     wpercent = float(basewidth / 512)
#     hsize = int(512 * wpercent)
#     img1 = img1.resize((basewidth, hsize), Image.ANTIALIAS)
#     img2 = img2.resize((basewidth, hsize), Image.ANTIALIAS)
#     img3 = img3.resize((basewidth, hsize), Image.ANTIALIAS)
#     # -------------------------
#     img1 = np.array(img1)
#     img2 = np.array(img2)
#     img3 = np.array(img3)
#     # -------------------------
#     img_array=cv2.merge([img1, img2, img3])
#     # -------------------------
#     X_SevereCytokine2700_224x224_3d_RGB.append(img_array)
# random.shuffle(X_SevereCytokine2700_224x224_3d_RGB)
# x_SevereCytokine2700_224x224_3d_RGB=[]
# for features in X_SevereCytokine2700_224x224_3d_RGB:
#     x_SevereCytokine2700_224x224_3d_RGB.append(features)
# x_SevereCytokine2700_224x224_3d_RGB=np.array(x_SevereCytokine2700_224x224_3d_RGB).reshape(-1,IMG_SIZE,IMG_SIZE,3)
# target_folder='/Volumes/Seagate Exp/CellImagingMainProject/'
# str=target_folder + 'X_SevereCytokine2700_224x224_3d_RGB.npy'
# np.save(str, x_SevereCytokine2700_224x224_3d_RGB, allow_pickle=True)

# #######################################################################################
# #######################################################################################

# target_folder='/Volumes/Seagate Exp/CellImagingMainProject/'
# str=target_folder + 'X_Healthy_512x512_3d_RGB.npy'
# X_Healthy_512x512_3d_RGB=np.load(str, allow_pickle=True)


# X_Healthy_224x224_3d_RGB=[]


# for i in range(len(X_Healthy_512x512_3d_RGB)):
#     basewidth = IMG_SIZE
#     img1=Image.fromarray((X_Healthy_512x512_3d_RGB[i,:,:,0]).astype(np.uint8))
#     img2=Image.fromarray((X_Healthy_512x512_3d_RGB[i,:,:,1]).astype(np.uint8))
#     img3=Image.fromarray((X_Healthy_512x512_3d_RGB[i,:,:,2]).astype(np.uint8))
#     # -------------------------
#     wpercent = float(basewidth / 512)
#     hsize = int(512 * wpercent)
#     img1 = img1.resize((basewidth, hsize), Image.ANTIALIAS)
#     img2 = img2.resize((basewidth, hsize), Image.ANTIALIAS)
#     img3 = img3.resize((basewidth, hsize), Image.ANTIALIAS)
#     # -------------------------
#     img1 = np.array(img1)
#     img2 = np.array(img2)
#     img3 = np.array(img3)
#     # -------------------------
#     img_array=cv2.merge([img1, img2, img3])
#     # -------------------------
#     X_Healthy_224x224_3d_RGB.append(img_array)
# random.shuffle(X_Healthy_224x224_3d_RGB)
# x_Healthy_224x224_3d_RGB=[]
# for features in X_Healthy_224x224_3d_RGB:
#     x_Healthy_224x224_3d_RGB.append(features)
# x_Healthy_224x224_3d_RGB=np.array(x_Healthy_224x224_3d_RGB).reshape(-1,IMG_SIZE,IMG_SIZE,3)
# target_folder='/Volumes/Seagate Exp/CellImagingMainProject/'
# str=target_folder + 'X_Healthy_224x224_3d_RGB.npy'
# np.save(str, x_Healthy_224x224_3d_RGB, allow_pickle=True)

# #######################################################################################
# #######################################################################################

# target_folder='/Volumes/Seagate Exp/CellImagingMainProject/'
# str=target_folder + 'X_NoCytokine_512x512_3d_RGB.npy'
# X_NoCytokine_512x512_3d_RGB=np.load(str, allow_pickle=True)


# X_NoCytokine_224x224_3d_RGB=[]


# for i in range(len(X_NoCytokine_512x512_3d_RGB)):
#     basewidth = IMG_SIZE
#     img1=Image.fromarray((X_NoCytokine_512x512_3d_RGB[i,:,:,0]).astype(np.uint8))
#     img2=Image.fromarray((X_NoCytokine_512x512_3d_RGB[i,:,:,1]).astype(np.uint8))
#     img3=Image.fromarray((X_NoCytokine_512x512_3d_RGB[i,:,:,2]).astype(np.uint8))
#     # -------------------------
#     wpercent = float(basewidth / 512)
#     hsize = int(512 * wpercent)
#     img1 = img1.resize((basewidth, hsize), Image.ANTIALIAS)
#     img2 = img2.resize((basewidth, hsize), Image.ANTIALIAS)
#     img3 = img3.resize((basewidth, hsize), Image.ANTIALIAS)
#     # -------------------------
#     img1 = np.array(img1)
#     img2 = np.array(img2)
#     img3 = np.array(img3)
#     # -------------------------
#     img_array=cv2.merge([img1, img2, img3])
#     # -------------------------
#     X_NoCytokine_224x224_3d_RGB.append(img_array)
# random.shuffle(X_NoCytokine_224x224_3d_RGB)
# x_NoCytokine_224x224_3d_RGB=[]
# for features in X_NoCytokine_224x224_3d_RGB:
#     x_NoCytokine_224x224_3d_RGB.append(features)
# x_NoCytokine_224x224_3d_RGB=np.array(x_NoCytokine_224x224_3d_RGB).reshape(-1,IMG_SIZE,IMG_SIZE,3)
# target_folder='/Volumes/Seagate Exp/CellImagingMainProject/'
# str=target_folder + 'X_NoCytokine_224x224_3d_RGB.npy'
# np.save(str, x_NoCytokine_224x224_3d_RGB, allow_pickle=True)
