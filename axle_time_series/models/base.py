import os
import shutil

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from progress.bar import Bar
import matplotlib.pyplot as plt

import dataset
import visualizations
import evaluation
import losses

class BraidModel:
    def __init__(self, dir_braid, name):
        self._dir_braid = dir_braid
        self._name = name
        self._model = None

        self._dir_models = f'{dir_braid}models/'
        self._dir_training = f'{dir_braid}training/{name}/'
        self._dir_evaluation = f'{dir_braid}evaluation/{name}/'
        self._dir_results = f'{dir_braid}results/{name}/'
        self._dir_plots = f'{dir_braid}plots/{name}/'

        if not os.path.exists(self._dir_models):
            os.makedirs(self._dir_models)

    def load(self):
        print('Loading', self._name)
        if self._model is not None:
            fname = f'{self._dir_models}{self._name}.weights.h5'
            if os.path.exists(fname):
                self._model.load_weights(fname)
            else:
                return False
        else:
            raise NotImplementedError    
        
        return True
    
    def save(self):
        print('Saving', self._name)
        if self._model is not None:
            self._model.save_weights(f'{self._dir_models}{self._name}.weights.h5')
        else:
            raise NotImplementedError    
    
    def train(self, X, Y):
        if os.path.exists(self._dir_training):
            shutil.rmtree(self._dir_training)
        os.makedirs(self._dir_training)
    
    def evaluate(self, X, Y, meta):
        if os.path.exists(self._dir_evaluation):
            shutil.rmtree(self._dir_evaluation)
        os.makedirs(self._dir_evaluation)

        # if os.path.exists(self._dir_results):
        #     shutil.rmtree(self._dir_results)
        # os.makedirs(self._dir_results)

        if not os.path.exists(self._dir_results):
            os.makedirs(self._dir_results)

        if os.path.exists(self._dir_plots):
            shutil.rmtree(self._dir_plots)
        os.makedirs(self._dir_plots)
    
    def print(self):
        if self._model is not None:
            print(self._name)
            print(self._model.summary())
