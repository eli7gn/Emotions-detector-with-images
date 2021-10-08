import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold

import keras
from keras.models import Model
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler, TerminateOnNaN, LambdaCallback
from keras.layers import Input

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

np.save("emotions_labels.npy", emotion_table)



# List of folders for training, validation and test.
#train_folders = ['C://Users//Eli7_Saxe//Documents//DSPG//vgg8-emotions//data//FER2013Train']
#valid_folders = ['C://Users//Eli7_Saxe//Documents//DSPG//vgg8-emotions//data//FER2013Valid'] 
test_folders  = ['C://Users//Eli7_Saxe//Documents//DSPG//vgg8-emotions//data//FER2013Test']

# read FER+ dataset.
#logging.info("Loading data...")
####

test_and_val_params = FERPlusParameters(num_classes, 48, 48, "majority", True)
test_data_reader    = FERPlusReader.create('data', test_folders, "label.csv", test_and_val_params)
#
X_test = np.load("emotions_arcface_vgg8_x_test.pny") 
y_test = np.load ("emotions_arcface_vgg8_arg_max_y_test.npy")
y_test0 = y_test

########
#model = keras.models.load_model("arcface_emotion_model.HDF5", custom_objects={'ArcFace': ArcFace})
#score = model.evaluate([X_test, y_test], y_test, verbose=1)
#arcface_model = Model(inputs=[input], outputs= arcface_model.output)
model1 = keras.models.load_model("arcface_emotion_copy.HDF5", custom_objects={'ArcFace': ArcFace})
for i, layer in enumerate(model1.layers):
    if i<19:
        layer.name = "layer" +str(i)
    
model1.save("arcface_emotion_copy.HDF5")       
        
model = keras.models.load_model("emotions_vgg8_3d_model.HDF5")
for i, layer in enumerate(model.layers):
    if i<19:
        layer.name = "layer" +str(i)

model.load_weights("arcface_emotion_copy.HDF5", by_name= True)
#model.save("emotions_arcface_load_weight_model")

k_pred = model.predict(X_test, verbose=1)
#k_pred = model.predict(X_test)


arg_max_k_pred = np.argmax(k_pred, axis=1)
#arg_max_y_test = np.argmax(y_test,axis=1)


np.save("emotions_arcface_vgg8_inference_output.npy", arg_max_k_pred)
#np.save("emotions_arcface_vgg8_scores_out.npy", scores)

