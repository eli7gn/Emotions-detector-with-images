# -*- coding: utf-8 -*-

"""
Created on Sun Dec  1 11:22:31 2019
@author: Eli7_Saxe
"""

import onnxruntime as rt
import numpy as np

X_test = np.load("emo_vgg8_3d_x_test.npy")

sess = rt.InferenceSession("emo_vgg8_ONNX.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
prob = pred[0]
print(prob.ravel()[:10])
