import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import csv

def feature_extraction_from_pretrained_model(model, pathname_file):
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    .......
    .....
    .....
    save as csv.
    return dataframe 

feature_extraction_from_pretrained_model(dataframe ) -- > next step

# -----------------------------------------------------------------