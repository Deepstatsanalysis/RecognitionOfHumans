import pandas as pd
#from keras.utils import np_utils
from keras.utils import to_categorical
import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import resizeImage as img
import crop

early_stopping = EarlyStopping(monitor='value_loss', patience=1)
humans = []
answers = []
'''
root_dir = 'humans/'

imagesPaths = glob.glob(os.path.join(root_dir, '*.jpg'))

for imagePath in imagesPaths:
    crop.resize(imagePath)
'''
root_dir = 'train_only_humans_ready/'

imagesPaths = glob.glob(os.path.join(root_dir, '*.jpg'))

for imagePath in imagesPaths:
    image = cv2.imread(imagePath)
    humans.append(image)

humans = np.asarray(humans)

print(humans.shape)

humans = np.array(humans, dtype="float32") / 255.0

fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(20, 10))
axes = axes.flat

for i in range(18):
    axes[i].imshow(humans[i])
    axes[i].axis('off')

answers = pd.read_csv(root_dir + 'train.csv', delimiter=';')['class']
print(answers)

X_train = humans
print(X_train)

y_train = np.asarray(answers)

#y_train = np_utils.to_categorical(y_train)
y_train= to_categorical(y_train, 2)
print(y_train)

print(X_train.shape)

X_test = []
y_test = []
root_dir = 'test_only_humans_ready/'

imagesPaths = glob.glob(os.path.join(root_dir, '*.jpg'))

for imagePath in imagesPaths:
    image = cv2.imread(imagePath)
    X_test.append(image)

X_test = np.array(X_test, dtype="float32") / 255.0

y_test = pd.read_csv(root_dir + 'test.csv', delimiter=';')['class']

y_test = np.asarray(y_test)

y_test= to_categorical(y_test, 2)

def get_model():
    model = Sequential()  # Создание модели

    model.add(Conv2D(32, (3, 3), input_shape=(294, 122, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = get_model()
model.summary()
print(X_train.shape)
print(y_train.shape)
model.fit(X_train, y_train, epochs=10, batch_size=5,validation_data=(X_test, y_test),
          callbacks=[ModelCheckpoint('models/modelBestKeksi.h5', save_best_only=True)])
