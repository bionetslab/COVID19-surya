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
from skimage.io import imread_collection

IMG_SIZE=32

directory='/Users/surya/Downloads/kagglecatsanddogs_3367a (1)/PetImages'
classes=['Cat', 'Dog']

training_data = []


def LoadImagesAndLabels():
    c_0=0
    c_1=0
    for c in classes:
        path=os.path.join(directory,c) # path to "0" or "1" directory
        class_num=classes.index(c)
        print(class_num)
        print(type(class_num))
    
        for filename in os.listdir(path):
            if class_num==0:
                c_0=c_0+1
                if not c_0>500:
                    fpath = os.path.join(path, filename)
                    fobj = open(fpath, "rb")
                    is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(100)
                    if is_jfif:
                        img = cv2.imread(os.path.join(path,filename))
                    if img is not None:
                        img=cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                        img=img.tolist()
                        training_data.append([img,class_num])        
            elif class_num==1:
                c_1=c_1+1
                if not c_1>500:
                    fpath = os.path.join(path, filename)
                    fobj = open(fpath, "rb")
                    is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(100)
                    if is_jfif:
                        img = cv2.imread(os.path.join(path,filename))
                    if img is not None:
                        img=cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                        img=img.tolist()
                        training_data.append([img,class_num]) 
    # return training_data
LoadImagesAndLabels()
# print(len(training_data))

random.shuffle(training_data)


X=[]
y=[]

for features, label in training_data:
    X.append(features)
    y.append(label)

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

# Save the training data
pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()
# Save the training labels
pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

##########################################################################################################

# X=pickle.load(open("X.pickle","rb"))
# y=pickle.load(open("y.pickle","rb"))

# # Normalize data
# X=np.asarray(X)/255.0
# # X=X.tolist()

# y = np.array(y)

# model=Sequential()

# model.add(  Conv2D(64,(3,3),input_shape=X.shape[1:])  )
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(  Conv2D(32,(2,2))    )
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Flatten())

# model.add(Dense(128))
# model.add(Activation('sigmoid'))

# # model.add(Dense(64))
# # model.add(Activation('sigmoid'))

# model.add(Dense(1))
# model.add(Activation("sigmoid"))

# model.compile(loss="binary_crossentropy",
#               optimizer="adam",
#               metrics=['accuracy'])

# # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# # history=model.fit(X, y, batch_size=32, shuffle=True, sample_weight=None, epochs=50,validation_split=0.1, verbose = 1, callbacks=[callback]) # seed=100,         

# history=model.fit(X, y, batch_size=32, shuffle=True, sample_weight=None, epochs=50,validation_split=0.1, verbose = 1) # seed=100,         


# # model.fit(X,y,batch_size=32,epochs=25,validation_split=0.1)

# # list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()






