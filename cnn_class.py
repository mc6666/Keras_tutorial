import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
import os

# 取得 MNIST 資料
def getData():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    img_rows, img_cols = 28, 28

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # CNN 需加一維
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    return X_train/255, y_train, X_test/255, y_test

# 訓練模型
def trainModel(X_train, y_train, X_test, y_test):
    batch_size = 64
    epochs = 15

    model = tf.keras.models.Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(10, activation='softmax'))

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1)

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    datagen.fit(X_train)
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=epochs,
                                  validation_data=datagen.flow(X_test, y_test, batch_size=batch_size), verbose=2,
				                  steps_per_epoch=X_train.shape[0]//batch_size)

    model.save('mnist_model.h5')
    return model
    
# 載入模型
def loadModel():
    return tf.keras.models.load_model('mnist_model.h5')

