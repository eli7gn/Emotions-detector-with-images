# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 11:03:28 2019

@author: Eli7_Saxe
"""
#import tensorflow as tf
import tf2onnx
#import tensorflow_core.tools.graph_transforms as graph_transforms
import keras
import onnx
import keras2onnx
#tf.compat.v1.disable_v2_behavior() 
model = keras.models.load_model('emotions_vgg8_3d_model.HDF5')
onnx_model = keras2onnx.convert_keras(model)
onnx.save_model(onnx_model, 'emo_vgg8_ONNX.onnx')