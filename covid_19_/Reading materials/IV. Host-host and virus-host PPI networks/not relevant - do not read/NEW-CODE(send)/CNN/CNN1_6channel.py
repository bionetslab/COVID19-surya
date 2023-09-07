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

y = genfromtxt('/Users/surya/Desktop/CellImagingRxRx/RxRx19b/metadata disease condition-first100.csv')
y = y.astype(int)
from sklearn.preprocessing import LabelBinarizer
# Transform labels to one-hot
# lb = LabelBinarizer()
# y = lb.fit_transform(y)

DATADIR="/Users/surya/Desktop/CellImagingRxRx/"
CATEGORIES=["RxRx19b_NPYs-first100"]

# for category in CATEGORIES:
#     path=os.path.join(DATADIR,category) # path to "0" or "1" directory
#     for img in os.listdir(path):
#         img_array=np.load(os.path.join(path,img))
        
#         img_array=np.asarray(img_array)
#         plt.imshow(img_array[:,:,1],cmap='gray',vmin=0,vmax=255)
#         plt.show()
#         # print(img_array)
#         # print(img_array.shape)

# # Reshaping the images
# IMG_SIZE=50
# new_array=cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# plt.imshow(new_array, cmap='gray')
# plt.show()

IMG_SIZE=128



# Create training data
training_data=[]

def create_training_data():
        for category in CATEGORIES:
            path=os.path.join(DATADIR,category) # path to "0" or "1" directory
            for i in range(1,101):
                img=path+'/'+str(i)+'.npy'
                print(img)
                img_array=np.load(os.path.join(path,img))
                img_array=np.asarray(img_array)
                img_array1=img_array[:,:,0]
                img_array2=img_array[:,:,1]
                img_array3=img_array[:,:,2]
                img_array4=img_array[:,:,3]
                img_array5=img_array[:,:,4]
                img_array6=img_array[:,:,5]
                
                
                basewidth = IMG_SIZE
                img1 = Image.fromarray(img_array1)
                img2 = Image.fromarray(img_array2)
                img3 = Image.fromarray(img_array3)
                img4 = Image.fromarray(img_array4)
                img5 = Image.fromarray(img_array5)
                img6 = Image.fromarray(img_array6)
                
                
                wpercent = float(basewidth / 1024)
                hsize = int(1024 * wpercent)
                img1 = img1.resize((basewidth, hsize), Image.ANTIALIAS)
                img2 = img2.resize((basewidth, hsize), Image.ANTIALIAS)
                img3 = img3.resize((basewidth, hsize), Image.ANTIALIAS)
                img4 = img4.resize((basewidth, hsize), Image.ANTIALIAS)
                img5 = img5.resize((basewidth, hsize), Image.ANTIALIAS)
                img6 = img6.resize((basewidth, hsize), Image.ANTIALIAS)
                # img.save('resized_image.jpg')
                
                img1 = np.array(img1)
                img2 = np.array(img2)
                img3 = np.array(img3)
                img4 = np.array(img4)
                img5 = np.array(img5)
                img6 = np.array(img6)
                
                img_array=cv2.merge([img1, img2, img3, img4, img5, img6])
                
                
                # img_array=img_array.reshape(IMG_SIZE,IMG_SIZE,6)
                # new_array=cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                # plt.imshow(img_array[:,:,1],cmap='gray',vmin=0,vmax=255)
                # plt.show()
                # print(img_array)
                # print(img_array.shape)
                training_data.append([img_array])
create_training_data()
print(len(training_data))

# Randomly shuffle the data
random.shuffle(training_data)

# Check for ten samples if the labels are correct
# for sample in training_data[:10]:
   #  print(sample[1])


X=[]

for features in training_data:
    X.append(features)
    # y.append(label)

X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,6)

# X=pickle.load(open("X.pickle","rb"))
# y=pickle.load(open("y.pickle","rb"))

# Normalize data
X=X/255.0

y = np.array(y)

model=Sequential()

model.add(  Conv2D(64,(3,3),input_shape=X.shape[1:])  )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(  Conv2D(64,(3,3))    )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('sigmoid'))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X,y,batch_size=32,epochs=3,validation_split=0.1)


