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
IMAGE_SIZE = (120, 120)
CATEGORIES = ['normal', 'virus', 'bacteria']


# OS path for data set
DATA_PATH = 'C:\data_set'

#Loading the data, normalization and later will add data augmentation

meta_data = pd.read_csv(os.path.join(DATA_PATH, 'metadata', 'chest_xray_metadata.csv'))

# Ovo je bio kod za razvrstavanje slika po folderima (klasama) za lakse ucitavanje i labele kasnije
# za 23 slike nisu nadjeni meta podaci, a 2 slike vezane za pusenje su takodje maknute

# for index, row in meta_data.iterrows():
#     print(row, ' je row')
#     if row.Label == 'Normal':
#         shutil.move(os.path.join(DATA_PATH, row.X_ray_image_name), os.path.join(DATA_PATH, CATEGORIES[0], row.X_ray_image_name))
#     elif row.Label_1_Virus_category == 'Virus':
#         shutil.move(os.path.join(DATA_PATH, row.X_ray_image_name),
#                     os.path.join(DATA_PATH, CATEGORIES[1], row.X_ray_image_name))
#     else:
#         shutil.move(os.path.join(DATA_PATH, row.X_ray_image_name),
#                     os.path.join(DATA_PATH, CATEGORIES[2], row.X_ray_image_name))

# ucitavanje slika pomocu image loadera iz kerasa
BATCH_SIZE = 64

# Funkcija je dostupna u tensor flow nighlty buildu.
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_PATH, 'data'),
    labels='inferred',
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_PATH, 'data'),
    validation_split=0.2,
    labels='inferred',
    subset="validation",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# dovodimo piksele u range 0 - 1


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

# koristimo advanced adam optimizator, da nam u prvoj iteraciji ne bude premala vrijednost zbog momenta koji je inicijalno 0
optimizer = tf.keras.optimizers.Adamax(learning_rate=LEARNING_RATE)

# koristimo SparseCategoricalCrossentropy jer cemo klase predstaviti kao integere, i imamo 3 klase
loss = keras.losses.CategoricalCrossentropy()


# Arhitektura resenja
model = tf.keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='random_normal'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3),strides=1, padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),strides=1, padding='same', activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='softmax'))
model.add(layers.Dense(3))

model.compile(optimizer='adamax',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(train_ds, ' je train')
print(val_ds, ' je val')


history = model.fit(train_ds, epochs=20,
                    validation_data=val_ds)

test_loss, test_acc = model.evaluate(val_ds, verbose=2)

model.summary()
