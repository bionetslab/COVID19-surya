

import matplotlib.pyplot as plt
import sys
import csv
import os
from PIL import Image
import numpy as np
import sys
import csv
import os
import pandas as pd
import numpy as np
from numpy import save
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


X=np.load('X_rxrx1_images_40000toend.npy')
y=np.load('y___sirna_40000toend.npy')
# X = X/255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104, test_size=0.25, shuffle=True)