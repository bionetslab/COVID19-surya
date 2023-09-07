import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib
import os
import sys
import matplotlib.pyplot as plt

sys.path.append('/Users/surya/Desktop/CellImagingRxRx/rxrx1-utils/rxrx1-utils-main')
import rxrx.io as rio

y = rio.load_site_as_rgb(dataset='train', experiment='HUVEC-08',
                         plate=4, well='K09', site=2)

plt.figure(figsize=(8, 8))
plt.axis('off')

_ = plt.imshow(y)