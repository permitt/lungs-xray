import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt

LEARNING_RATE = 0.001
IMAGE_SIZE = 160
CATEGORIES = ['normal', 'virus', 'bacteria']


# OS path for data set
DATA_PATH = 'C:\data_set'

#Loading the data, normalization and later will add data augmentation

meta_data = pd.read_csv(os.path.join(DATA_PATH, 'metadata', 'chest_xray_metadata.csv'))


for index, row in meta_data.iterrows():
    print(row, ' je row')
    if row.Label == 'Normal':
        shutil.move(os.path.join(DATA_PATH, row.X_ray_image_name), os.path.join(DATA_PATH, CATEGORIES[0], row.X_ray_image_name))
    elif row.Label_1_Virus_category == 'Virus':
        shutil.move(os.path.join(DATA_PATH, row.X_ray_image_name),
                    os.path.join(DATA_PATH, CATEGORIES[1], row.X_ray_image_name))
    else:
        shutil.move(os.path.join(DATA_PATH, row.X_ray_image_name),
                    os.path.join(DATA_PATH, CATEGORIES[2], row.X_ray_image_name))


model = tf.keras.models.Sequential()
model.add(layers.Conv2D())
