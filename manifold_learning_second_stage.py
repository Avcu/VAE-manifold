# this script learns the manifold using regular auto encoder and variational auto encoder 
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
sys.path.append('D:/Codes-Main-Python/variational_inference/VAE_manifold/utils')
sys.path.append('D:/Codes-Main-Python/variational_inference/VAE_manifold/plotting_utils')
import numpy as np
from scipy import stats, io
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from betavae_methods import bvae_model
from gammavae_methods import gvae_model
from manifold_methods import manifold_static
from dataset_methods import data_set_model 
from plotting_methods import *
import tadasets


##% this whole section gets (loads) the datapoints and latent factors from the manifold
options_true_manifold = {}
options_true_manifold['manifold_type'] = 'distorted_cylinder'
options_true_manifold['num_points'] = 15000
options_true_manifold['noise_value'] = 0
manifold_train_main = manifold_static(options_true_manifold)
manifold_train_main.generate_manifold()
points_train = manifold_train_main.points_manifold

options_true_manifold = {}
options_true_manifold['manifold_type'] = 'distorted_cylinder'
options_true_manifold['num_points'] = 4000
options_true_manifold['noise_value'] = 0
options_true_manifold['fixed_values'] = [0,0.25,0.5,1]#[np.pi/8, np.pi/4, 2*np.pi/4, 3*np.pi/4, 7*np.pi/8]#[0,np.pi]
manifold_test = manifold_static(options_true_manifold)
manifold_test.generate_manifold()
points_test = manifold_test.points_manifold

dataset_manifold = data_set_model(x_train = points_train, x_test = points_test, options = [])
manifold_train_main.plot_manifold()
# create bvae options
options = {}
options['dim_latent'] = 2 # dimension of latent variabels
options['activation_type'] = 'relu' # activation function in the layers 'tanh' or None or 'relu' or 'elu'
options['learning_rate'] = 1e-3 # what is learning rate (usually 0.001 to 0.01)
options['num_epochs'] = 400 # number of epochs
options['batch_size'] = 400 # number of batches in trainig
options['iteration_plot_show'] = 1000 # when to plot the results
options['print_info_step'] = 50 # how many steps to show info
options['sw_plot'] = 1 # switch for plotting the results within trials
options['plot_mode'] = 'save' #'save' or 'show'
options['optimizer_type'] = 'Adam' # 'Adam' or 'GD'
options['make_latent_cycle'] = 1
options['train_restore'] = 'restore' # 'train' or 'resotre'
options['dropout_rate'] = 0.3
options['manifold_train'] = manifold_train_main # if you put it as empty it will be disregarded
options['manifold_test'] = manifold_test # if you put it as empty it will be disregarded
options['save_global_directory'] = 'D:/Codes-Main-Python/variational_inference/VAE_manifold/results'
options['beta_or_gamma'] = 'gamma' # 'beta' or 'gamma'
options['gen_linear'] = 0 # 1: for linear generative network, 0 for nonlinear

if options['beta_or_gamma'] == 'beta': # to use bvae
    main_model = bvae_model(data_set = dataset_manifold, beta = 0.01, options = options)

elif options['beta_or_gamma'] == 'gamma': # to use gvae
    #capacity_schedule = np.concatenate( ( np.zeros((5))  ,  np.linspace(0,25,options['num_epochs'] - 20) , 25 * np.ones((15)) )  )
    capacity_schedule = np.concatenate( ( np.zeros((10))  ,  np.linspace(0,0,options['num_epochs'] - 10) )  )
    #capacity_schedule = np.concatenate( ( np.linspace(0,5,options['num_epochs'] - 300), 5 * np.ones((300)) )  )
    main_model = gvae_model(data_set = dataset_manifold, gamma = 0, capacity_schedule = capacity_schedule, options = options)

file_path = "{}/saved_files/datapoints.mat".format(main_model.save_main_folder)

loaded_datapoints = io.loadmat(file_path)

latent_mu = loaded_datapoints['z_mu']
latent_log_sigma2 = loaded_datapoints['z_log_sigma2']
x_true = loaded_datapoints['x']
x_recons = loaded_datapoints['x_recons']
points_colors = loaded_datapoints['points_colors']
##% now prepare them for the second VAE (this time aiming for disentnaglement)
options_true_manifold = {}
options_true_manifold['manifold_type'] = 'from_latents'
options_true_manifold['num_points'] = np.shape(latent_mu)[0]
options_true_manifold['noise_value'] = 0
manifold_train = manifold_static(options_true_manifold)
manifold_train.points_manifold = latent_mu
manifold_train.points_colors = points_colors
points_train = manifold_train.points_manifold

##% now we run the second stage vae
dataset_manifold = data_set_model(x_train = points_train, x_test = points_train, options = [])
manifold_train.plot_manifold()
##% create bvae options
options = {}
options['dim_latent'] = np.shape(latent_mu)[1] # dimension of latent variabels
options['activation_type'] = 'relu' # activation function in the layers 'tanh' or None or 'relu' or 'elu'
options['learning_rate'] = 1e-3 # what is learning rate (usually 0.001 to 0.01)
options['num_epochs'] = 400 # number of epochs
options['batch_size'] = 400 # number of batches in trainig
options['iteration_plot_show'] = 1000 # when to plot the results
options['print_info_step'] = 50 # how many steps to show info
options['sw_plot'] = 1 # switch for plotting the results within trials
options['plot_mode'] = 'save' #'save' or 'show'
options['optimizer_type'] = 'Adam' # 'Adam' or 'GD'
options['make_latent_cycle'] = 1
options['train_restore'] = 'train' # 'train' or 'resotre'
options['dropout_rate'] = 0.3
options['manifold_train'] = manifold_train # if you put it as empty it will be disregarded
options['manifold_test'] = manifold_train # if you put it as empty it will be disregarded
options['save_global_directory'] = main_model.save_main_folder
options['save_main_folder'] = '{}/trained_model'.format(main_model.save_main_folder)
options['beta_or_gamma'] = 'gamma' # 'beta' or 'gamma'
options['gen_linear'] = 0 # 1: for linear generative network, 0 for nonlinear

##% create beta or gamma vae
if options['beta_or_gamma'] == 'beta': # to use bvae
    bvaemodel = bvae_model(data_set = dataset_manifold, beta = 0.01, options = options)
    if options['train_restore'] is 'train':
        bvaemodel.build_computation_graph()
        bvaemodel.train_model()
    elif options['train_restore'] is 'restore':
        bvaemodel.load_model(epoch_num = 500)
        bvaemodel.plot_test(batch_x = points_test, manifold = manifold_test)
elif options['beta_or_gamma'] == 'gamma': # to use gvae
    #capacity_schedule = np.concatenate( ( np.zeros((5))  ,  np.linspace(0,25,options['num_epochs'] - 20) , 25 * np.ones((15)) )  )
    capacity_schedule = np.concatenate( ( np.zeros((10))  ,  np.linspace(0,0,options['num_epochs'] - 10) )  )
    #capacity_schedule = np.concatenate( ( np.linspace(0,5,options['num_epochs'] - 200), 5 * np.ones((200)) )  )
    gvaemodel = gvae_model(data_set = dataset_manifold, gamma = 1, capacity_schedule = capacity_schedule, options = options)
    if options['train_restore'] is 'train':
        gvaemodel.build_computation_graph()
        gvaemodel.train_model()
    elif options['train_restore'] is 'restore':
        gvaemodel.load_model(epoch_num = 400)
        gvaemodel.plot_test(batch_x = points_train, manifold = manifold_train)
        gvaemodel.save_datapoints_manifold()