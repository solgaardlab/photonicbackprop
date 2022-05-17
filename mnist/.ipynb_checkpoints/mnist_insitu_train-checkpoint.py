import os
from layers import TM_taps
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

parser = argparse.ArgumentParser(description='mnist_insitu_train')
parser.add_argument('--p_error',
                    default=0.0,
                    type=float,
                    help='field phase error')
parser.add_argument('--a_error',
                    default=0.0,
                    type=float,
                    help='field amplitude error')
parser.add_argument('--poiss_K',
                    default=0.0,
                    type=float,
                    help='poisson shot noise factor')
parser.add_argument('--num_epochs',
                    default=5,
                    type=int,
                    help='num epochs')
parser.add_argument('--learning_rate',
                    default=0.001,
                    type=float,
                    help='learning rate')
args = parser.parse_args()

def get_final_pred(output, num_classes):
    pred = (output[:,:num_classes])**2
    return tf.math.l2_normalize(pred)

@tf.function
def get_loss(output, y, num_classes):
    pred = get_final_pred(output, num_classes)
    loss = tf.math.reduce_sum((pred - y)**2)
    return loss

def final_stage(output, y, num_classes):
    B, F = output.shape
    output = tf.Variable(tf.math.abs(output))
    with tf.GradientTape() as tape:
        loss = get_loss(output, y, num_classes) * np.sqrt(num_classes)
    grad = tf.cast(tape.gradient(loss, output), dtype = tf.complex64)
    return output, grad, loss.numpy()

def extract_gradients_from_powers_normalize(forward_p, backward_p, sum_p, \
                                             forward_mag, backward_mag, sum_mag):
    gradients = tf.reduce_mean((sum_p*sum_mag - forward_p*forward_mag - backward_p*backward_mag) / 2, axis = 0)
    ## average in mini-batch
    return gradients[::2,::2], gradients[1::2,::2]

def get_noisy_input(inp, a_error_in, p_error_in):
    inp = tf.math.l2_normalize(inp, axis = -1)
    return inp * tf.cast((1 + a_error_in * tf.random.normal(shape = inp.shape, dtype = tf.float32)), dtype = tf.complex64) * \
        tf.math.exp(1j*p_error_in*np.pi*tf.cast(tf.random.normal(shape = inp.shape, dtype = tf.float32), dtype = tf.complex64))

def get_noisy_meas(out, a_error_out, p_error_out):
    return out * tf.cast((1 + a_error_out * tf.random.normal(shape = out.shape, dtype = tf.float32)), dtype = tf.complex64) * \
        tf.math.exp(1j*p_error_out*np.pi*tf.cast(tf.random.normal(shape = out.shape, dtype = tf.float32), dtype = tf.complex64))

def insitu_backprop_simu_noise_normalize(model, optimizer, x, y, num_classes, poiss_K, \
                               a_error_in, p_error_in, a_error_out, p_error_out):
    ## forward pass
    pred = dict()
    pred['input_0'] = x
    
    for ll in range(len(model)):
        pred[f'forward_mag_{ll}'] = tf.math.abs(tf.norm(pred[f'input_{ll}'], ord='euclidean', axis = -1))
        model_in = get_noisy_input(pred[f'input_{ll}'], a_error_in, p_error_in)
        pred[f'forward_{ll}'] = model[ll].transform(model_in)
        pred[f'forward_meas_{ll}'] = get_noisy_meas(pred[f'forward_{ll}'][...,-1], a_error_out, p_error_out)
        pred[f'forward_meas_{ll}'] *= tf.cast(pred[f'forward_mag_{ll}'], tf.complex64)
        pred[f'input_{ll + 1}'] = tf.cast(tf.math.abs(pred[f'forward_meas_{ll}']), tf.complex64)
        
    final_pred, pred[f'adjoint_{len(model)}'], loss = final_stage(pred[f'input_{ll + 1}'], y, num_classes)
    
    ## backward pass
    for ll in range(len(model))[::-1]:
        pred[f'error_{ll}'] = \
        tf.complex(tf.math.real(pred[f'adjoint_{ll + 1}'])*tf.math.cos(tf.math.angle(pred[f'forward_meas_{ll}'])), \
                    -tf.math.real(pred[f'adjoint_{ll + 1}'])*tf.math.sin(tf.math.angle(pred[f'forward_meas_{ll}'])))
        
        pred[f'backward_mag_{ll}'] = tf.math.abs(tf.norm(pred[f'error_{ll}'], ord='euclidean', axis = -1))
        model_in = get_noisy_input(pred[f'error_{ll}'], a_error_in, p_error_in)
        pred[f'backward_{ll}'] = model[ll].inverse_transform(model_in)[...,::-1]
        pred[f'backward_meas_{ll}'] = get_noisy_meas(pred[f'backward_{ll}'][...,0], a_error_out, p_error_out)
        pred[f'backward_meas_{ll}'] *= tf.cast(pred[f'backward_mag_{ll}'], tf.complex64)
        pred[f'adjoint_{ll}'] = pred[f'backward_meas_{ll}']
    
    ## sum pass
    for ll in range(len(model)):
        pred[f'sum_mag_{ll}'] = \
        tf.math.abs(tf.norm(pred[f'input_{ll}'] - 1j * tf.math.conj(pred[f'adjoint_{ll}']), ord='euclidean', axis = -1))
        model_in = get_noisy_input(pred[f'input_{ll}'] - 1j * tf.math.conj(pred[f'adjoint_{ll}']), a_error_in, p_error_in)
        pred[f'sum_{ll}'] = model[ll].transform(model_in)
    
    ## gradient estimation
    grad_list = []
    for ll in range(len(model)):
        forward_p = tf.transpose(tf.math.abs(pred[f'forward_{ll}'][...,1:])**2, perm = [0,2,1])
        forward_p += tf.random.normal(shape = forward_p.shape, dtype = tf.float32) * np.sqrt(forward_p * poiss_K)
        backward_p = tf.transpose(tf.math.abs(pred[f'backward_{ll}'][...,1:])**2, perm = [0,2,1])
        backward_p += tf.random.normal(shape = backward_p.shape, dtype = tf.float32) * np.sqrt(backward_p * poiss_K)
        sum_p = tf.transpose(tf.math.abs(pred[f'sum_{ll}'][...,1:])**2, perm = [0,2,1])
        sum_p += tf.random.normal(shape = sum_p.shape, dtype = tf.float32) * np.sqrt(sum_p * poiss_K)
        
        grad_theta, grad_phi = \
            extract_gradients_from_powers_normalize(forward_p, backward_p, sum_p, \
            pred[f'forward_mag_{ll}']**2, pred[f'backward_mag_{ll}']**2, pred[f'sum_mag_{ll}']**2)
        
        grad_list.append(grad_theta)
        grad_list.append(grad_phi)
    
    var_list = []
    for ll in range(len(model)):
        var_list.append(model[ll].theta)
        var_list.append(model[ll].phi)
    optimizer.apply_gradients(zip(grad_list, var_list))
    
    pred['gradient'] = grad_list
    return loss, pred
    

if __name__ == "__main__":
    a_error = args.a_error
    p_error = args.p_error
    poiss_K = args.poiss_K
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    
    mnist_dp = MNISTDataProcessor()
    data_N64 = mnist_dp.fourier(4)
    
    num_layers = 2
    N = 64
    num_classes = N
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    log_dir = 'mnist_trainlog_bs1_lr{}_aerror{}_perror{}_poissK{}_{}'.format(learning_rate, a_error, \
                  p_error, poiss_K, datetime.now().strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    model = []
    for ll in range(num_layers):
        model.append(TM_taps(N, hadamard = False, theta_init_name='haar_tri', \
                             phi_init_name='random_phi', name=f'onn_layer{ll}'))
        model[ll].gamma = tf.zeros_like(model[ll].gamma)
    
    for ll in range(num_layers):
        np.save(os.path.join(log_dir, 'theta{}_insitu_init.npy'.format(ll)),\
                model[ll].theta.numpy())
        np.save(os.path.join(log_dir, 'phi{}_insitu_init.npy'.format(ll)),\
                model[ll].phi.numpy())
        np.save(os.path.join(log_dir, 'gamma{}_insitu_init.npy'.format(ll)),\
                model[ll].gamma.numpy())
    
    error_kwargs = {}
    error_kwargs['poiss_K'] = poiss_K
    error_kwargs['a_error_in'] = a_error
    error_kwargs['p_error_in'] = p_error
    error_kwargs['a_error_out'] = a_error
    error_kwargs['p_error_out'] = p_error
    
    train_loss = []
    for ee in range(num_epochs):
        for idx in tqdm(range(data_N64.x_train.shape[0])):
            x = tf.convert_to_tensor(data_N64.x_train[idx:(idx+1)])
            y = tf.convert_to_tensor(data_N64.y_train[idx:(idx+1)].astype(np.float32))
            y = tf.concat([y, tf.zeros(shape = (1,N-10), dtype = tf.float32)], axis = -1)
            
            loss, pred = insitu_backprop_simu_noise_normalize(model, optimizer, x, y, num_classes, **error_kwargs)
            train_loss.append(loss)
            with open(os.path.join(log_dir, 'train_loss_insitu_all.txt'), 'a+') as f:
                f.write('epoch:{}, idx:{}, loss:{}\n'.format(ee, idx, loss))
        
        for ll in range(num_layers):
            np.save(os.path.join(log_dir, 'theta{}_insitu_epoch_{}.npy'.format(ll, ee)),\
                    model[ll].theta.numpy())
            np.save(os.path.join(log_dir, 'phi{}_insitu_epoch_{}.npy'.format(ll, ee)),\
                    model[ll].phi.numpy())
            np.save(os.path.join(log_dir, 'gamma{}_insitu_epoch_{}.npy'.format(ll, ee)),\
                    model[ll].gamma.numpy())

    train_loss = np.asarray(train_loss)
    np.save(os.path.join(log_dir, 'train_loss_insitu_all.npy'), train_loss)


    