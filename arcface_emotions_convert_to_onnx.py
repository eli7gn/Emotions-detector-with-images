# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 11:03:28 2019

@author: Eli7_Saxe
"""
#import tensorflow as tf
import tf2onnx
import keras
import onnx
import keras2onnx
from keras.utils import CustomObjectScope
from metrics import ArcFace

with  CustomObjectScope({'ArcFace': ArcFace}):
    model = keras.models.load_model('arcface_emotion_model.HDF5')
    onnx_model = keras2onnx.convert_keras(model)
    onnx.save_model(onnx_model, 'arcface_emotion_ONNX.onnx')