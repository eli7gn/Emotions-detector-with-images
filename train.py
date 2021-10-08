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

arch_names = archs.__dict__.keys()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='vgg8_sphereface',
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg8_sphereface',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: vgg8_sphereface)')
    parser.add_argument('--num-features', default=3, type=int,
                        help='dimention of embedded features')
    parser.add_argument('--scheduler', default='CosineAnnealing',
                        choices=['CosineAnnealing', 'None'],
                        help='scheduler: ' +
                            ' | '.join(['CosineAnnealing', 'None']) +
                            ' (default: CosineAnnealing)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min-lr', default=1e-3, type=float,
                        help='minimum learning rate')
    parser.add_argument('--momentum', default=0.5, type=float)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # add model name to args
    args.name = 'emotions_%s_%dd' %(args.arch, args.num_features)

    os.makedirs('models/%s' %args.name, exist_ok=True)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)
#######################################################################################
#######################################################################################
    emotion_table = {'neutral'  : 0, 
                     'happiness': 1, 
                     'surprise' : 2, 
                     'sadness'  : 3, 
                     'anger'    : 4, 
                     'disgust'  : 5, 
                     'fear'     : 6, 
                     'contempt' : 7}
    num_classes = len(emotion_table)
    
    # List of folders for training, validation and test.
    train_folders = ['C://Users//Eli7_Saxe//Documents//DSPG//vgg8-emotions//data//FER2013Train']
    valid_folders = ['C://Users//Eli7_Saxe//Documents//DSPG//vgg8-emotions//data//FER2013Valid'] 
    test_folders  = ['C://Users//Eli7_Saxe//Documents//DSPG//vgg8-emotions//data//FER2013Test']

    # read FER+ dataset.
    #logging.info("Loading data...")
    train_params        = FERPlusParameters(num_classes, 48, 48, 'majority', False)
    test_and_val_params = FERPlusParameters(num_classes, 48, 48, "majority", True)

    train_data_reader   = FERPlusReader.create('data', train_folders, "label.csv", train_params)
    val_data_reader     = FERPlusReader.create('data', valid_folders, "label.csv", test_and_val_params)
    test_data_reader    = FERPlusReader.create('data', test_folders, "label.csv", test_and_val_params)
    
    # print summary of the data.
    display_summary(train_data_reader, val_data_reader, test_data_reader)

    
    X, y, current_batch_size = train_data_reader.next_minibatch(1)
    X_test, y_test, current_batch_size_test =  test_data_reader.next_minibatch(1)
    
    X = np.moveaxis(X, 1, 3) 
    X_test = np.moveaxis(X_test, 1, 3)


    # (X, y), (X_test, y_test) = mnist.load_data()

    # X = X[:, :, :, np.newaxis].astype('float32') / 255
    # X_test = X_test[:, :, :, np.newaxis].astype('float32') / 255

#    y = keras.utils.to_categorical(y, num_classes)
#    y_test = keras.utils.to_categorical(y_test, num_classes)

#######################################################################################
#######################################################################################
    if args.optimizer == 'SGD':
        optimizer = SGD(lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = Adam(lr=args.lr)

    model = archs.__dict__[args.arch](args)
    model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
    model.summary()

    callbacks = [
        ModelCheckpoint(os.path.join('models', args.name, 'model.hdf5'),
            verbose=1, save_best_only=True,  save_weights_only=False),
        CSVLogger(os.path.join('models', args.name, 'log.csv')),
        TerminateOnNaN()]

    if args.scheduler == 'CosineAnnealing':
        callbacks.append(CosineAnnealingScheduler(T_max=args.epochs, eta_max=args.lr, eta_min=args.min_lr, verbose=1))

    if 'face' in args.arch:
        # callbacks.append(LambdaCallback(on_batch_end=lambda batch, logs: print('W has nan value!!') if np.sum(np.isnan(model.layers[-4].get_weights()[0])) > 0 else 0))
        model.fit([X, y], y, validation_data=([X_test, y_test], y_test),
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1)
    else:
        model.fit(X, y, validation_data=(X_test, y_test),
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1)
    model.save(args.name+"_model.hdf5")
    model.load_weights(os.path.join('models/%s/model.hdf5' %args.name))
    if 'face' in args.arch:
        score = model.evaluate([X_test, y_test], y_test, verbose=1)
    else:
        score = model.evaluate(X_test, y_test, verbose=1)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


if __name__ == '__main__':
    main()
