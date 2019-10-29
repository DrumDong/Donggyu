# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:54:15 2019

@author: ehdrb
"""

import tensorflow as tf

tf.__version__

import keras

keras.__version__


from keras.utils import np_utils

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Activation

 
#######Mnist 예제#############################################

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784).astype('float32') / 255.0

X_test = X_test.reshape(10000, 784).astype('float32') / 255.0

Y_train = np_utils.to_categorical(Y_train)

Y_test = np_utils.to_categorical(Y_test)

 

model = Sequential()

model.add(Dense(units=64, input_dim=28*28, activation='relu'))

model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=5, batch_size=32)

 

loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

 

print('loss_and_metrics : ' + str(loss_and_metrics))