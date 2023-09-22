


from PIL import Image
import numpy as np
import sys
import csv
import os
import pandas as pd
import numpy as np

image = Image.open('/home/surya/Documents/GitHub/COVID19-surya/covid_19_/orig_file.png').convert('L')
image2 = Image.open('/home/surya/Documents/GitHub/COVID19-surya/covid_19_/orig_file.png').convert('L')
# image.show()






# # The file format of the source file.
# print(image.format) # Output: JPEG

# # The pixel format used by the image. Typical values are "1", "L", "RGB", or "CMYK."
# print(image.mode) # Output: RGB

# # Image size, in pixels. The size is given as a 2-tuple (width, height).
# print(image.size) # Output: (1920, 1280)

# # Colour palette table, if any.
# print(image.palette) # Output: None

# # # Chnging image type:
# # image.save('new_image.png')

new_image = image.resize((128, 128))
new_image2 = image2.resize((128, 128))

numpy_image = np.array(new_image)
numpy_image2 = np.array(new_image2)


image_=np.dstack([numpy_image, numpy_image2])
# new_image.save('image_400.jpg')

# print(image.size) # Output: (1920, 1280)
# print(new_image.size) # Output: (400, 400)

# image.thumbnail((400, 400))
# image.save('image_thumbnail.jpg')
# print(image.size) # Output: (400, 267)