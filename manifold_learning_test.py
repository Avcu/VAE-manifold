# this script learns the manifold using regular auto encoder and variational auto encoder 
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
sys.path.append('D:/Codes-Main-Python/variational_inference/tests/utils')
sys.path.append('D:/Codes-Main-Python/variational_inference/tests/plotting_utils')
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from betavae_methods import bvae_model
from manifold_methods import manifold_static
from dataset_methods import data_set_model 
from plotting_methods import *
import tadasets


# give the data points to learn the representations using vae
# to test
#landmarks = tadasets.torus(n=1000,c=2,a=1)
options_true_manifold = {}
options_true_manifold['manifold_type'] = 'cylinder'
options_true_manifold['num_points'] = 15000
manifold_train = manifold_static(options_true_manifold)
manifold_train.generate_manifold()
points_train = manifold_train.points_manifold

#
options_true_manifold['num_points'] = 10000
manifold_test = manifold_static(options_true_manifold)
manifold_test.generate_manifold()
points_test = manifold_test.points_manifold
#
#points_test = tadasets.dsphere(n=3000, d=1, r=3, noise = 0.1)

dataset_manifold = data_set_model(x_train = points_train, x_test = points_test, options = [])
manifold_train.plot_manifold()
# create bvae options
options = {}
options['dim_latent'] = 2 # dimension of latent variabels
options['activation_type'] = 'relu' # activation function in the layers 'tanh' or None or 'relu' or 'elu'
options['learning_rate'] = 1e-3 # what is learning rate (usually 0.001 to 0.01)
options['num_epochs'] = 400 # number of epochs
options['batch_size'] = 500 # number of batches in trainig
options['iteration_plot_show'] = 1000 # when to plot the results
options['print_info_step'] = 50 # how many steps to show info
options['sw_plot'] = 1 # switch for plotting the results within trials
options['plot_mode'] = 'save' #'save' or 'show'
options['optimizer_type'] = 'Adam' # 'Adam' or 'GD'
options['make_latent_cycle'] = 1
options['train_restore'] = 'train'
options['dropout_rate'] = 0.3
options['manifold_train'] = manifold_train # if you put it as empty it will be disregarded
options['manifold_test'] = manifold_test # if you put it as empty it will be disregarded
options['save_global_directory'] = 'D:/Codes-Main-Python/variational_inference/tests/results'
options['z_var'] = 1
# to use bvae
bvaemodel = bvae_model(data_set = dataset_manifold, beta = 0, options = options)
if options['train_restore'] is 'train':
    bvaemodel.build_computation_graph()
    bvaemodel.train_model()
elif options['train_restore'] is 'restore':
    bvaemodel.load_model(epoch_num = 200)
    bvaemodel.plot_test(batch_x = points_test)
# # to use gvae
# #capacity_schedule = np.concatenate( ( np.zeros((5))  ,  np.linspace(0,25,options['num_epochs'] - 20) , 25 * np.ones((15)) )  )
# capacity_schedule = np.concatenate( ( np.zeros((10))  ,  np.linspace(0,25,options['num_epochs'] - 10) )  )
# gvaemodel = gvae_model(data_set = data_dspr, gamma = 10, capacity_schedule = capacity_schedule, options = options)
# if options['train_restore'] is 'train':
#   gvaemodel.build_computation_graph()
#   gvaemodel.train_model()
# elif options['train_restore'] is 'restore':
#   gvaemodel.load_model(epoch_num = 120)
#   batch_0 = np.asarray( np.squeeze(data_dspr.test_interpolate[:,0,:,:]) , dtype = 'float32')
#   batch_1 = np.asarray( np.squeeze(data_dspr.test_interpolate[:,1,:,:]) , dtype = 'float32')
#   gvaemodel.plot_latent_interpolate(batch_0,batch_1)