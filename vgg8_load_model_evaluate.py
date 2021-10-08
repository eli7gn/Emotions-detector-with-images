#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:01:13 2017

"""

import keras
from keras.models import Model
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#a = np.asarray([0,1,2,3,4])
#>>> a == 0 # or whatver
#array([ True, False, False, False, False], dtype=bool)

x_test = x_test.astype('float32')
x_test = x_test[:, :, :, np.newaxis].astype('float32') / 255
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.models.load_model("vgg8_model.HDF5")
scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
k_pred = model.predict(x_test)


arg_max_k_pred = np.argmax(k_pred, axis=1)
arg_max_y_test = np.argmax(y_test,axis=1)


np.save("vgg8_minst_inference_output.npy", arg_max_k_pred)
np.save("vgg8_minst_scores_out.npy", scores)

np.save("vgg8_minst_arg_max_y_test.npy", arg_max_y_test)
np.save("vgg8_minst_x_test.npy",x_test)

#u1:
#a=np.load("kcifar10_argmax_y_test.npy")
b=arg_max_k_pred.astype('u1')
np.save("u1_vgg8_minst__infer_out.npy",b)

b=arg_max_y_test.astype('u1')
np.save("u1_vgg8_minst_y_test.npy",b)



print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
