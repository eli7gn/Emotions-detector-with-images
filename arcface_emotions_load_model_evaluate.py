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
import archs
from metrics import *
from scheduler import *
import archs
from metrics import *
from scheduler import *
from ferplus import *


emotion_table = {'neutral'  : 0, 
                 'happiness': 1, 
                 'surprise' : 2, 
                 'sadness'  : 3, 
                 'anger'    : 4, 
                 'disgust'  : 5, 
                 'fear'     : 6, 
                 'contempt' : 7}

num_classes = len(emotion_table)
emotion_table = np.array(emotion_table)

test_folders  = ['C://Users//Eli7_Saxe//Documents//DSPG//vgg8-emotions//data//FER2013Test']
test_and_val_params = FERPlusParameters(num_classes, 48, 48, "majority", True)
test_data_reader    = FERPlusReader.create('data', test_folders, "label.csv", test_and_val_params)
x_test, y_test, current_batch_size_test =  test_data_reader.next_minibatch(1)

x_test = np.moveaxis(x_test, 1, 3)

model = keras.models.load_model("arcface_emotion_model.HDF5", custom_objects={'ArcFace': ArcFace})
scores = model.evaluate([x_test, y_test], y_test, batch_size=128, verbose=1)
k_pred = model.predict(x_test)



arg_max_k_pred = np.argmax(k_pred, axis=1)
arg_max_y_test = np.argmax(y_test,axis=1)


np.save("emo_arcface_3d_inference_output.npy", arg_max_k_pred)
np.save("emo_arcface_scores_out.npy", scores)

np.save("emo_arcface_arg_max_y_test.npy", arg_max_y_test)
np.save("emo_arcface_x_test.npy",x_test)

#u1:
#a=np.load("kcifar10_argmax_y_test.npy")
b=arg_max_k_pred.astype('u1')
np.save("u1_emo_arcface_3d_infer_out.npy",b)

b=arg_max_y_test.astype('u1')
np.save("u1_emo_arcface_3d_y_test.npy",b)



print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
