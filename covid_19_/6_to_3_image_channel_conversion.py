import matplotlib.pyplot as plt

import sys


'''
if 'google.colab' in sys.modules:
    !git clone https://github.com/recursionpharma/rxrx1-utils
    sys.path.append('/content/rxrx1-utils')
'''


sys.path.append('rxrx1-utils')
import rxrx.io as rio
# C:\ProgramData\Anaconda3\Lib\site-packages\rxrx
import csv
import os


with open('RxRx1 dataset/metadata.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    counter = 0  # Separate counter variable to control the number of iterations
   # t = 0
    for line in csv_reader:
        
        # if counter >= 2:
          #  break  # Break the loop after 2 iterations
        
        try:
            t = rio.load_site('images', line['experiment'], line['plate'], line['well'], line['site'])
            # t.shape
            x = rio.convert_tensor_to_rgb(t)
            # x.shape


            # For visualize all six channels at once
            print(f"experiment {line['experiment']}, plate {line['plate']}, wall {line['well']}, site {line['site']}: {x.shape}")

            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, 
            right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.figure(figsize=(10, 10))
            plt.axis('off')
            plt.imshow(x)
            filename = f"{line['experiment']}_plate_{line['plate']}_wall_{line['well']}_site_{line['site']}.png"
            plt.savefig(f'D:/PROJECT_BIOMEDICAL/6_to_3_Image/output/{filename}', bbox_inches = 'tight', pad_inches = 0)
            plt.close()
            
            counter += 1  # Increment the counter variable
            print(counter)
            
        except FileNotFoundError as e:
            print(f'Error processing image: Experiment {line["experiment"]}, Plate {line["plate"]}, Well {line["well"]}, Site {line["site"]}')
            print(str(e))
            continue
        
        except Exception as e:
            print(f'Error processing image: Experiment {line["experiment"]}, Plate {line["plate"]}, Well {line["well"]}, Site {line["site"]}')
            print(str(e))
            continue
        
