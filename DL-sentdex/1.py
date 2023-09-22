import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from matplotlib import pyplot
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import load
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle
from tensorflow.keras import optimizers
import sys

# # ===============================================================================
# # ===============================================================================
# # ===============================================================================

# # # =================== Pre-processing and saving as npy: ===================


# # DATADIR = "kagglecatsanddogs_5340/"
# # CATEGORIES = ["Dog", "Cat"]
# # for category in CATEGORIES:  # do dogs and cats
# #     path = os.path.join(DATADIR,category)  # create path to dogs and cats
# #     all_images_in_path=os.listdir(path)
# #     cnt=-1
# #     for img in all_images_in_path:  # iterate over each image per dogs and cats
# #         cnt+=1
# #         if cnt>9:
# #             break
# #         # img_array = cv2.imread(os.path.join(path,img))  # convert to array
# #         img_array = imread(os.path.join(path,img))
# #         # plt.subplot(330 + 1 + cnt)
# #         plt.imshow(img_array)  # graph it
# #         plt.show()  # display!

# # -------------------------------------------------------------


# DATADIR = "kagglecatsanddogs_5340/"
# CATEGORIES = ["Dog", "Cat"]
# X, y = list(), list()

# CATEGORIES_labels_dict=dict(zip(CATEGORIES, [0,1]))

# problematic_images=[]

# for category in CATEGORIES:  # do dogs and cats
#     path = os.path.join(DATADIR,category)  # create path to dogs and cats
#     all_images_in_path=os.listdir(path)
#     cnt=-1
#     for img in all_images_in_path:  # iterate over each image per dogs and cats
#         cnt+=1
#         print(img)
#         # if cnt>200:
#         #     break
#         # # # img_array = cv2.imread(os.path.join(path,img))  # convert to array
#         # # img_array = imread(os.path.join(path,img))
#         # # # plt.subplot(330 + 1 + cnt)
#         # # plt.imshow(img_array)  # graph it
#         # # plt.show()  # display!
#         # load image:
#         try:
#             photo = load_img(os.path.join(path, img), target_size=(200, 200))
#             # convert to numpy array
#             photo = img_to_array(photo)
#             # store
#             X.append(photo)
#             y.append(CATEGORIES_labels_dict[category])
#         except:
#             problematic_images.append(img)
            
        
# # convert to a numpy arrays
# X = asarray(X)
# y = asarray(y)
# # print(X.shape, y.shape)
# # save the reshaped photos
# save('dogs_vs_cats_photos.npy', X)
# save('dogs_vs_cats_labels.npy', y)

# # --------------------------------------------------------------




# ===============================================================================
# ===============================================================================
# ===============================================================================

# =================== Loading npy: ===================

# load and confirm the shape
X = load('dogs_vs_cats_photos.npy')[0:1000]
y = load('dogs_vs_cats_labels.npy')[0:1000]
# print(X.shape, y.shape)
X = X/255.0

# ---------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104, test_size=0.25, shuffle=True)
  
# # printing out train and test sets:
# print('X_train : ')
# print(X_train)
# print('')
# print('X_test : ')
# print(X_test)
# print('')
# print('y_train : ')
# print(y_train)
# print('')
# print('y_test : ')
# print(y_test)


# ===============================================================================
# ===============================================================================
# ===============================================================================

# # =================== Define model: ===================

# ----------- 1. VGG (# hidden layers=1) -----------
        
# define cnn model
def define_model():
    model = Sequential()
    # model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=X.shape[1:]))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
 	# compile model
    opt = optimizers.SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# # # ----------- 2. (ConvPool1 with relu and MaxPool, ConvPool2 with relu and MaxPool, Flatten, Dense, Dense with sigmoid; adam optimizer; metrics=['accuracy']): -----------                                                              

# # define cnn model
# def define_model():
#     model = Sequential()
#     model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
    
#     model.add(Conv2D(256, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
    
#     model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    
#     model.add(Dense(64))
    
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
    
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model



# ----------- 1. VGG (# hidden layers=1) -----------
        
# define cnn model
def define_vgg_model(n):
    if n==1:
        model = Sequential()
        # model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=X.shape[1:]))
        model.add(MaxPooling2D((2, 2)))
        # # ----- 20% Dropout: -----
        # model.add(Dropout(0.2))
        # # -------------------------
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        # # ----- 50% Dropout: -----
        # model.add(Dropout(0.5))
        # # -------------------------
        model.add(Dense(1, activation='sigmoid'))
     	
    elif n==2:
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
        model.add(MaxPooling2D((2, 2)))
        # # ----- 20% Dropout: -----
        # model.add(Dropout(0.2))
        # # -------------------------
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        # # ----- 20% Dropout: -----
        # model.add(Dropout(0.2))
        # # -------------------------
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        # # ----- 50% Dropout: -----
        # model.add(Dropout(0.5))
        # # -------------------------
        model.add(Dense(1, activation='sigmoid'))
        
    elif n==3:
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
        model.add(MaxPooling2D((2, 2)))
        # ----- 20% Dropout: -----
        model.add(Dropout(0.2))
        # -------------------------
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        # ----- 20% Dropout: -----
        model.add(Dropout(0.2))
        # -------------------------
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        # ----- 20% Dropout: -----
        model.add(Dropout(0.2))
        # -------------------------
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        # ----- 50% Dropout: -----
        model.add(Dropout(0.5))
        # -------------------------
        model.add(Dense(1, activation='sigmoid'))
 
    # compile model
    opt = optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# ===============================================================================
# ===============================================================================
# ===============================================================================

# # =================== Summarize diagnostics: ===================

# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    # pyplot.close()


# ===============================================================================
# ===============================================================================
# ===============================================================================

# # =================== Run: ===================


# # define model
# model = define_model()
# # print(model)


# define model
n=3 # no. of blocks
model = define_vgg_model(n)
# print(model)


# fit model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=5, verbose=1)
# model.fit_generator(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=20, verbose=1)
# print(history)


# evaluate model
_, acc = model.evaluate(X_test, steps=len(X_test), verbose=0)
print('> %.3f' % (acc * 100.0))


summarize_diagnostics(history)


# ==============================================================================================================
# ==============================================================================================================
# ============================================================================================================== 
# ==============================================================================================================


# =================================================
# EARLY STOPPING (EarlyStopping callback in Keras):
# =================================================



















