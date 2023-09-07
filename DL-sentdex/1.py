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

# DATADIR = "kagglecatsanddogs_5340/"
# CATEGORIES = ["Dog", "Cat"]
# for category in CATEGORIES:  # do dogs and cats
#     path = os.path.join(DATADIR,category)  # create path to dogs and cats
#     all_images_in_path=os.listdir(path)
#     cnt=-1
#     for img in all_images_in_path:  # iterate over each image per dogs and cats
#         cnt+=1
#         if cnt>9:
#             break
#         # img_array = cv2.imread(os.path.join(path,img))  # convert to array
#         img_array = imread(os.path.join(path,img))
#         # plt.subplot(330 + 1 + cnt)
#         plt.imshow(img_array)  # graph it
#         plt.show()  # display!

# =============================================================


DATADIR = "kagglecatsanddogs_5340/"
CATEGORIES = ["Dog", "Cat"]
X, y = list(), list()

CATEGORIES_labels_dict=dict(zip(CATEGORIES, [0,1]))

for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    all_images_in_path=os.listdir(path)
    cnt=-1
    for img in all_images_in_path:  # iterate over each image per dogs and cats
        cnt+=1
        if cnt>9:
            break
        # # # img_array = cv2.imread(os.path.join(path,img))  # convert to array
        # # img_array = imread(os.path.join(path,img))
        # # # plt.subplot(330 + 1 + cnt)
        # # plt.imshow(img_array)  # graph it
        # # plt.show()  # display!
        # load image:
        photo = load_img(os.path.join(path, img), target_size=(200, 200))
        # convert to numpy array
        photo = img_to_array(photo)
        # store
        X.append(photo)
        y.append(CATEGORIES_labels_dict[category])
        # convert to a numpy arrays
X = asarray(X)
y = asarray(y)
print(X.shape, y.shape)
# save the reshaped photos
save('dogs_vs_cats_photos.npy', X)
save('dogs_vs_cats_labels.npy', y)

# ===============================================================================

# load and confirm the shape
X = load('dogs_vs_cats_photos.npy')
y = load('dogs_vs_cats_labels.npy')
print(X.shape, y.shape)
X = X/255.0

# ===============================================================================

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

# # ====================================== 1. VGG (# hidden layers=1) =========================================
        
# # define cnn model
# def define_model():
#     model = Sequential()
#     # model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
#     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=X.shape[1:]))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
#     model.add(Dense(1, activation='sigmoid'))
#  	# compile model
#     opt = optimizers.SGD(learning_rate=0.001, momentum=0.9)
#     model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
#     # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # define model
# model = define_model()

# # fit model
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=20, verbose=1)
# # model.fit_generator(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=20, verbose=1)




# # ====================================== 2. (ConvPool1 with relu and MaxPool, ConvPool2 with relu and MaxPool, Flatten, Dense, Dense with sigmoid; adam optimizer; metrics=['accuracy']): =========================================

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    
    model.add(Dense(64))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# define model
model = define_model()

# fit model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=20, verbose=1)
# model.fit_generator(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=20, verbose=1)




# # ====================================== 2. (ConvPool1 with relu and MaxPool, ConvPool2 with relu and MaxPool, Flatten, Dense, Dense with sigmoid; adam optimizer; metrics=['accuracy']): =========================================











# for category in CATEGORIES:  # do dogs and cats
#     path = os.path.join(DATADIR,category)  # create path to dogs and cats
#     for img in os.listdir(path):  # iterate over each image per dogs and cats
#         img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
#         plt.imshow(img_array, cmap='gray')  # graph it
#         plt.show()  # display!

#         break  # we just want one for now so break
#     break  #...and one more!



# # define location of dataset
# folder = 'kagglecatsanddogs_5340/'
# photos, labels = list(), list()
# # enumerate files in the directory
# for file in listdir(folder):
    
#     # determine class
#     output = 0.0
#     if file.startswith('dog'):
#         output = 1.0
#     # load image
#     photo = load_img(folder + file, target_size=(200, 200))
#     # convert to numpy array
#     photo = img_to_array(photo)
#     # store
#     photos.append(photo)
#     labels.append(output)
#     # convert to a numpy arrays
#     photos = asarray(photos)
#     labels = asarray(labels)
#     print(photos.shape, labels.shape)
#     # save the reshaped photos
#     save('dogs_vs_cats_photos.npy', photos)
#     save('dogs_vs_cats_labels.npy', labels)

























# # ================================

# for category in CATEGORIES:  # do dogs and cats
#     path = os.path.join(DATADIR,category)  # create path to dogs and cats
#     for img in os.listdir(path):  # iterate over each image per dogs and cats
#         img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
#         plt.imshow(img_array, cmap='gray')  # graph it
#         plt.show()  # display!

#         break  # we just want one for now so break
#     break  #...and one more!

# # ================================

# print(img_array)
# print(img_array.shape)

# # ================================

# # IMG_SIZE = 50
# IMG_SIZE = 100

# new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# plt.imshow(new_array, cmap='gray')
# plt.show()

# # ================================
# # ================================

# training_data = []

# def create_training_data():
#     for category in CATEGORIES:  # do dogs and cats

#         path = os.path.join(DATADIR,category)  # create path to dogs and cats
#         class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

#         for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
#             try:
#                 img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
#                 new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
#                 training_data.append([new_array, class_num])  # add this to our training_data
#             except Exception as e:  # in the interest in keeping the output clean...
#                 pass
#             #except OSError as e:
#             #    print("OSErrroBad img most likely", e, os.path.join(path,img))
#             #except Exception as e:
#             #    print("general exception", e, os.path.join(path,img))

# create_training_data()

# print(len(training_data))

# # ================================

# import random

# random.shuffle(training_data)

# for sample in training_data[:10]:
#     print(sample[1])

# # ================================

# X = []
# y = []

# for features,label in training_data:
#     X.append(features)
#     y.append(label)

# print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# # ================================

# import pickle

# pickle_out = open("X.pickle","wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open("y.pickle","wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()

# # ================================

# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)

# pickle_in = open("y.pickle","rb")
# y = pickle.load(pickle_in)

# # ================================















































