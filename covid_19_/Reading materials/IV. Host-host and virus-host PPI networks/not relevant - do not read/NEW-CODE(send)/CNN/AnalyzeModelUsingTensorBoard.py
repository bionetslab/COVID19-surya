import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras. preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import time

#name model
NAME="RxRx1-Approx600Samples-64x2-{}".format(int(time.time()))

#define tensorboard callback
tensorboard=TensorBoard(log_dir='logs/{}'.format(NAME))

#gpu options
gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

pickle_in=open("X.pickle","rb")
X=pickle.load(pickle_in)

pickle_in=open("y.pickle","rb")
y=pickle.load(pickle_in)

X=X/255.0

y = np.array(y)

model=Sequential()

model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# model.add(Dense(64))
# model.add(Activation('sigmoid'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X,y,batch_size=32,epochs=20,validation_split=0.1,callbacks=[tensorboard]) #include list of all callbacks, here we only have tensorboard
