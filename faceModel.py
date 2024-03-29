from __future__ import print_function
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential, model_from_yaml, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from tensorflow.keras.optimizers import Adam, SGD
from keras.utils import np_utils, plot_model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2

num_classes = 2
img_rows, img_cols = 235, 235
batch_size = 16

from keras.preprocessing.image import ImageDataGenerator
def load(path):
    files = []
    images = []
    labels = []
    check = True
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            files.append(file)
    for file in files:
        img_array = cv2.imread(path+"/"+file)
        new_array = cv2.resize(img_array, (img_rows, img_cols))
        images.append(new_array)
        print(file)
        
        if 'Cagri' in file:
            labels.append(0)
        elif 'Yahya' in file:
            labels.append(1)

    images = np.array(images)
    images = np.array(images).reshape(-1, img_rows, img_cols, 3)
    labels = np.array(labels)
    labels = np_utils.to_categorical(labels, 2)
    return images,labels


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(img_rows, img_cols, 3), activation='relu', padding='same'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.summary()

sgd = Adam(lr=0.0003)
model.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['accuracy'])

images,labels = load("./faces2")
images = images.astype('float32')
images = images/255
print(images.shape,labels.shape)
for i in range(0,200):
    print(labels[i])
model.fit(images, labels, batch_size=batch_size, epochs=8, verbose=1, validation_split=0.1,shuffle=True)

# Evaluating the model
print("evaluate......")
score, accuracy = model.evaluate(images[:3], labels[:3], batch_size=batch_size)
print('score:', score, 'accuracy:', accuracy)
model.save('family.model')
