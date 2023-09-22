

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


# ---
Line=[]
rxrx1_path="/home/surya/Documents/AIBE-Surya-PhD/rxrx1/HUVEC/images/"
no_of_channels=6
# ---
rxrx1_images=[]
sirnas=[]
sirna_ids=[]

# ---
# with open('/home/surya/Documents/AIBE-Surya-PhD/rxrx1/HUVEC/metadata_huvec.csv', 'r') as csv_file:
csv_=pd.read_csv('/home/surya/Documents/AIBE-Surya-PhD/rxrx1/HUVEC/metadata_huvec.csv')
csv_=csv_.iloc[55000:]
csv_.to_csv('csv_.csv')
with open('csv_.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    counter = 0  # Separate counter variable to control the number of iterations
   # t = 0
    for line in csv_reader:
        # ---
        print(line)
        Line.append(line)
        filepath=""
        # ===
        try:
            img=[]
            for i in range(1,no_of_channels+1):
                filepath+='/'+line['experiment']+'/'+'Plate'+str(line['plate'])+'/'+  ( line['experiment'] + '_' + 's' + str(line['plate']) + '_' + 'w' + str(i) + '.png' )                                              
                sirna=line['sirna']
                sirna_id=line['sirna_id']
                # ---
                image = Image.open('/home/surya/Documents/GitHub/COVID19-surya/covid_19_/orig_file.png').convert('L')
                image=image.resize((128, 128))
                numpy_image = np.array(image)
                numpy_image=numpy_image/255.0
                img.append(numpy_image)
            rxrx1_images.append(np.dstack(np.array(img)))
            sirnas.append(sirna)
            sirna_ids.append(sirna_id)
        # ---
        except FileNotFoundError as e:
            print(f'Error processing image: Experiment {line["experiment"]}, Plate {line["plate"]}, Well {line["well"]}, Site {line["site"]}')
            print(str(e))
            continue
        # ---
        except Exception as e:
            print(f'Error processing image: Experiment {line["experiment"]}, Plate {line["plate"]}, Well {line["well"]}, Site {line["site"]}')
            print(str(e))
            continue
        # ===

# X=np.array(rxrx1_images)
# y___sirna=np.array(sirna)
# y___sirna_id=np.array(sirna_id)

X=rxrx1_images
y___sirna=sirnas
y___sirna_id=sirna_ids

save('X_rxrx1_images.npy', X)
save('y___sirna.npy', y___sirna)
save('y___sirna_id.npy', y___sirna_id)

# # ==========================
# n1=np.load('X_rxrx1_images_0to12000.npy')
# n2=np.load('X_rxrx1_images_12000to24000.npy')
# N=np.concatenate((n1, n2))
# save('X_rxrx1_images_0to24000.npy', N)
# # ==========================















