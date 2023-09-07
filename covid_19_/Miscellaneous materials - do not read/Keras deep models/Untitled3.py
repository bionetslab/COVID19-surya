import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
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
from sklearn.model_selection import train_test_split
import os 
import zipfile 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
import matplotlib.pyplot as plt


# folder='/Volumes/Seagate Exp/CellImagingMainProject/'
folder=''

# strng=folder+'X_SevereCytokine2700_Healthy_randomizedFinal_512x512_RGB.npy'
# # strng=folder+'X_SevereCytokine2700_Healthy_randomizedFinal_128x128.npy'

strng=folder+'X_SevereCytokine2700_Healthy_randomizedFinal_224x224_RGB.npy'
# strng=folder+'X_SevereCytokine260_NoCytokine_randomized.npy'


x=np.load(strng)
x=x/255.0
X=[]
for i in range(len(x)):
    X.append(x[i].tolist())

# strng=folder+'y_SevereCytokine2700_Healthy_randomizedFinal_512x512_RGB.npy'
# # strng=folder+'y_SevereCytokine2700_Healthy_randomizedFinal_128x128.npy'


strng=folder+'y_SevereCytokine2700_Healthy_randomizedFinal_224x224_RGB.npy'
# strng=folder+'y_SevereCytokine260_NoCytokine_randomized.npy'


y=np.load(strng)

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

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.30, random_state=1)

trainX = np.array(trainX)
testX = np.array(testX)
trainy = np.array(trainy)
testy = np.array(testy)
#####

# train_ds=[]
# test_ds=[]

# for i in range(len(trainy)):
#     train_ds.append([trainX[i],trainy[i]])

# for i in range(len(testy)):
#     test_ds.append([testX[i],testy[i]])


# data_augmentation = Sequential(
#     [
#         layers.RandomFlip("horizontal"),
#         layers.RandomRotation(0.1),
#     ]
# )

# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
#     for i in range(9):
#         augmented_images = data_augmentation(images)
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(augmented_images[0].numpy().astype("uint8"))
#         plt.axis("off")

#####

# from tensorflow.keras.applications.vgg16 import VGG16

# base_model = VGG16(input_shape = (224, 224, 3), # Shape of our images
# include_top = False, # Leave out the last fully connected layer
# weights = 'imagenet')

# for layer in base_model.layers:
#     layer.trainable = False

# # Flatten the output layer to 1 dimension
# x = layers.Flatten()(base_model.output)

# # Add a fully connected layer with 512 hidden units and ReLU activation
# x = layers.Dense(512, activation='relu')(x)

# # Add a dropout rate of 0.5
# x = layers.Dropout(0.5)(x)

# # Add a final sigmoid layer with 1 node for classification output
# x = layers.Dense(1, activation='sigmoid')(x)

# model = tf.keras.models.Model(base_model.input, x)

# model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])

# history=model.fit(trainX, trainy, batch_size=32, shuffle=True, sample_weight=None, epochs=50,validation_split=0.1, verbose = 1) # seed=100,

###################

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(224, 224, 3)))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])

model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss')

history=model.fit(trainX, trainy, batch_size=32, shuffle=True, sample_weight=None, epochs=50,validation_split=0.1, verbose = 1, callbacks=[callback])

###################

# model=Sequential()

# # model.add(  Conv2D(64,(3,3),input_shape=trainX.shape[1:])  )
# model.add(  Conv2D(64,(3,3),input_shape=np.shape(trainX)[1:])  )
# model.add(Activation("relu"))
# # model.add(  Conv2D(64,(3,3))    )
# # model.add(Activation("relu"))
# model.add(Dropout(0.5))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Activation('relu'))
# model.add(Dense(1))
# model.add(Activation("sigmoid"))
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# # summarize network:
# model.summary()

# # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss')


# history=model.fit(trainX, trainy, batch_size=32, shuffle=True, sample_weight=None, epochs=50,validation_split=0.1, verbose = 1, callbacks=[callback]) # seed=100,         

#################











# evaluate model (Generate generalization metrics):
score = model.evaluate(testX, testy, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()