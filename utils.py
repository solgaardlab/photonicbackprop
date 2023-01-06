import numpy as np
import scipy as sp

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Lambda
from keras import backend as K
from keras.datasets import mnist, fashion_mnist
from dataclasses import dataclass
import holoviews as hv

# from neurophox.ml.nonlinearities import cnormsq

# import seaborn as sns
from collections import namedtuple
import panel as pn
import xarray as xr

from holoviews.streams import Pipe
from typing import Optional, List, Dict

def norm_inputs(inputs, feature_axis=1):
    if feature_axis == 1:
        n_features, n_examples = inputs.shape
    elif feature_axis == 0:
        n_examples, n_features = inputs.shape
    for i in range(n_features):
        l1_norm = np.mean(np.abs(inputs[i, :]))
        inputs[i, :] /= l1_norm
    return inputs

ONNData = namedtuple('ONNData', ['x_train', 'y_train', 'y_train_ind', 'x_test', 'y_test', 'y_test_ind', 'units', 'num_classes'])

class MNISTDataProcessor:
    def __init__(self, fashion: bool=False):
        (self.x_train_raw, self.y_train), (self.x_test_raw, self.y_test) = fashion_mnist.load_data() if fashion else mnist.load_data()
        self.num_train = self.x_train_raw.shape[0]
        self.num_test = self.x_test_raw.shape[0]
        self.x_train_ft = np.fft.fftshift(np.fft.fft2(self.x_train_raw), axes=(1, 2))
        self.x_test_ft = np.fft.fftshift(np.fft.fft2(self.x_test_raw), axes=(1, 2))
        
    def fourier(self, freq_radius):
        min_r, max_r = 14 - freq_radius, 14 + freq_radius
        x_train_ft = self.x_train_ft[:, min_r:max_r, min_r:max_r]
        x_test_ft = self.x_test_ft[:, min_r:max_r, min_r:max_r]
        return ONNData(
            x_train=norm_inputs(x_train_ft.reshape((self.num_train, -1))).astype(np.complex64),
            y_train=np.eye(10)[self.y_train],
            y_train_ind=self.y_train,
            x_test=norm_inputs(x_test_ft.reshape((self.num_test, -1))).astype(np.complex64),
            y_test=np.eye(10)[self.y_test],
            y_test_ind=self.y_test,
            units=(2 * freq_radius)**2,
            num_classes=10
        )
    
    def resample(self, p, b=0):
        m = 28 - b * 2
        min_r, max_r = b, 28 - b
        x_train_ft = sp.ndimage.zoom(self.x_train_raw[:, min_r:max_r, min_r:max_r], (1, p / m, p / m))
        x_test_ft = sp.ndimage.zoom(self.x_test_raw[:, min_r:max_r, min_r:max_r], (1, p / m, p / m))
        return ONNData(
            x_train=norm_inputs(x_train_ft.reshape((self.num_train, -1)).astype(np.complex64)),
            y_train=np.eye(10)[self.y_train],
            x_test=norm_inputs(x_test_ft.reshape((self.num_test, -1)).astype(np.complex64)),
            y_test=np.eye(10)[self.y_test],
            units=p ** 2,
            num_classes=10
        )

@dataclass
class Metrics:
    metric_list: List[str]
    num_runs: int = 1
    logs: int = 0
    
    def update(self, val_dict: Dict[str, float]):
        for name, val in val_dict.items():
            self.metric_dict[name].update(val)
    
    def reset(self):
        for name in self.metric_dict:
            self.metric_dict[name].reset()
            
    def panel(self, props: Dict[str, dict]):
        def plot_metric(metric: xr.DataArray):
            return data
            
        return pn.Row(*[
            hv.Overlay([
                self.metric_dict[g].plot().opts(title=name, **props.get(g, {}))
                for g in group
            ]).collate()
            for name in self.metric_list
        ])

def metrics(names, xlabel='Epoch'):
    metric_dict = {}
    for name in names:
        metric_dict[name] = Metric(
            name,
            Pipe(),
            []
        )
    return Metrics(metric_dict)