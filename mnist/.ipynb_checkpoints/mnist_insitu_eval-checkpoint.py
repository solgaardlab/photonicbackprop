import os
from layers import TMTaps
from utils import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Activation, Lambda

import tensorflow as tf
import time
from tqdm import tqdm
import pdb
from datetime import datetime

import argparse
import os
    
parser = argparse.ArgumentParser(description='mnist_insitu_eval')
parser.add_argument('--log_dir',
                    default=None,
                    type=str,
                    help='model log path',
                    required=True)
parser.add_argument('--num_epochs',
                    default=30,
                    type=int,
                    help='epochs to eval')
args = parser.parse_args()

mnist_dp = MNISTDataProcessor()
data_N64 = mnist_dp.fourier(4)

def get_noisy_input_normalize(inp, a_error_in, p_error_in):
    inp = tf.math.l2_normalize(inp, axis = -1)
    return inp * tf.cast((1 + a_error_in * tf.random.normal(shape = inp.shape, dtype = tf.float32)), dtype = tf.complex64) * \
        tf.math.exp(1j*p_error_in*np.pi*tf.cast(tf.random.normal(shape = inp.shape, dtype = tf.float32), dtype = tf.complex64))

def get_noisy_meas_normalize(out, a_error_out, p_error_out):
    return out * tf.cast((1 + a_error_out * tf.random.normal(shape = out.shape, dtype = tf.float32)), dtype = tf.complex64) * \
        tf.math.exp(1j*p_error_out*np.pi*tf.cast(tf.random.normal(shape = out.shape, dtype = tf.float32), dtype = tf.complex64))

def get_final_pred_normalize(output, num_classes):
    pred = (output[:,:num_classes])**2
    return tf.math.l2_normalize(pred)

def insitu_inference_simu_noise_normalize(model, x, y, num_classes, poiss_K, \
                               a_error_in, p_error_in, a_error_out, p_error_out):
    ## forward pass
    pred = dict()
    pred['input_0'] = x
    
    for ll in range(len(model)):
        pred[f'forward_mag_{ll}'] = tf.math.abs(tf.norm(pred[f'input_{ll}'], ord='euclidean', axis = -1))
        model_in = get_noisy_input_normalize(pred[f'input_{ll}'], a_error_in, p_error_in)
        pred[f'forward_{ll}'] = model[ll].transform(model_in)
        pred[f'forward_meas_{ll}'] = get_noisy_meas_normalize(pred[f'forward_{ll}'][...,-1], a_error_out, p_error_out)
        pred[f'forward_meas_{ll}'] *= tf.cast(pred[f'forward_mag_{ll}'], tf.complex64)
        pred[f'input_{ll + 1}'] = tf.cast(tf.math.abs(pred[f'forward_meas_{ll}']), tf.complex64)
    
    final_pred = np.argmax(get_final_pred_normalize(pred[f'input_{ll + 1}'], num_classes).numpy())
    
    return final_pred == np.argmax(y.numpy())


if __name__ == "__main__":

    N = 64
    num_layers = 2
    num_epochs = args.num_epochs
    log_dir = args.log_dir
    
    error_kwargs = {}
    error_kwargs['poiss_K'] = 0.0
    error_kwargs['a_error_in'] = 0.0
    error_kwargs['p_error_in'] = 0.0
    error_kwargs['a_error_out'] = 0.0
    error_kwargs['p_error_out'] = 0.0

    test_acc = []
    for epoch in range(-1, num_epochs):
        model = []
        if epoch >= 0:
            for ll in range(num_layers):
                model.append(TMTaps(N, hadamard = False, theta_init_name='haar_tri', \
                             phi_init_name='random_phi', name=f'onn_layer{ll}'))
                model[ll].theta = tf.convert_to_tensor(np.load(os.path.join(log_dir, \
                                            'theta{}_insitu_epoch_{}.npy'.format(ll, epoch))))
                model[ll].phi = tf.convert_to_tensor(np.load(os.path.join(log_dir, \
                                            'phi{}_insitu_epoch_{}.npy'.format(ll, epoch))))
                model[ll].gamma = tf.convert_to_tensor(np.load(os.path.join(log_dir, \
                                            'gamma{}_insitu_epoch_{}.npy'.format(ll, epoch))))
        else:
            for ll in range(num_layers):
                model.append(TMTaps(N, hadamard = False, theta_init_name='haar_tri', \
                             phi_init_name='random_phi', name=f'onn_layer{ll}'))
                model[ll].theta = tf.convert_to_tensor(np.load(os.path.join(log_dir, \
                                            'theta{}_insitu_init.npy'.format(ll))))
                model[ll].phi = tf.convert_to_tensor(np.load(os.path.join(log_dir, \
                                            'phi{}_insitu_init.npy'.format(ll))))
                model[ll].gamma = tf.convert_to_tensor(np.load(os.path.join(log_dir, \
                                            'gamma{}_insitu_init.npy'.format(ll))))
        acc = 0
        num_test = 10000
        for idx in tqdm(range(num_test)):
            x = tf.convert_to_tensor(data_N64.x_test[idx:(idx+1)])
            y = tf.convert_to_tensor(data_N64.y_test[idx:(idx+1)].astype(np.float32))
            res = insitu_inference_simu_noise_normalize(model, x, y, N, **error_kwargs)
            if res:
                acc += 1
        test_acc.append(acc/num_test)
        print(acc/num_test)
    test_acc = np.asarray(test_acc)
    np.save(os.path.join(log_dir, 'test_acc_epochs_noerrorinf.npy'), test_acc)
