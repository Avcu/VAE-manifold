# this function is to test manifolds and auto encoders
#
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('D:/Codes-Main-Python/variational_inference/from_outside/kvae/simulation')
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from mpl_toolkits.mplot3d import Axes3D
from manifold_utils import manifold 
from var_autoencoder_methods import var_autoencoder_model, simple_autoencoder_model
# change seed
np.random.seed(42)
m = 5000
w1, w2 = 0.1, 0.3
noise = 0.05
manifold_shape = 'nonlinear' # 'linear' or 'nonlinear'
angles = np.random.rand(m) * 3 *np.pi /2 - 0.5
data = np.empty((m,3))

if manifold_shape is 'linear':
    data[:,0] = np.cos(angles) + np.sin(angles/2) + noise * np.random.randn(m)/2
    data[:,1] = np.sin(angles) * 0.7 + np.sin(angles/2) + noise * np.random.randn(m)/2
    data[:,2] = data[:,0] * w1 + data[:,1] * w2 + noise * np.random.randn(m)
else:
    #start building the manifold
    settings_manif = {}
    settings_manif['manifold_shape'] = 'infinity'
    settings_manif['num_samples_trial'] = 5000
    settings_manif['state_dim'] = 1
    settings_manif['state_A'] = 0.999
    settings_manif['state_Q'] = 0.1
    settings_manif['num_trials'] = 10
    settings_manif['step_size'] = 0.3
    settings_manif['state_type'] = 'additional'

    manif = manifold(settings_manif)
    manif.generate_z()
    manif.generate_m()
    print('manifold generated with these settings {}'.format(manif.settings))
    data = manif.m[0,:,:]
    data = np.transpose(data)
    print('shape data on manifold is {}'.format(data.shape))
    data[:,0] = data[:,0] + noise * np.random.randn(data.shape[0])
    data[:,1] = data[:,1] + noise * np.random.randn(data.shape[0])
    data[:,2] = data[:,2] + noise * np.random.randn(data.shape[0])
    
# normalize data
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
x_train = scalar.fit_transform(data)
x_test = np.empty(shape=[2,3])
x_test[0,:] = [1.2,1.2,0.4]
#x_test[1,:] = [-0.5,-0.5,-0.2]
x_test[1,:] = [-2,1.5,0]
# plot data
fig = plt.figure(figsize=[12,9])
ax = fig.add_subplot(111,projection='3d')
ax.scatter3D(x_train[:,0],x_train[:,1],x_train[:,2])
ax.scatter3D(x_test[:,0],x_test[:,1],x_test[:,2],s=200,marker='x',color='g',linewidth = 3)
ax.set_xlabel('dim 1')
ax.set_ylabel('dim 2')
ax.set_zlabel('dim 3')
ax.set_title('data points on manifold')
plt.pause(0.05)
plt.show()


# build var_autoencoder model


options_vae= {}

options_vae['dim_target'] = 3 # dimension of target signal
options_vae['dim_target_pre'] = 5 # dimension of encoder's layer before latent variabless
options_vae['dim_hidden'] = 1 # dimension of latent variabels
options_vae['dim_hidden_pre'] = 5 # dimension of hidden layer before latent variabless
options_vae['target_train'] = x_train# dimension of hidden layer before latent variabless
options_vae['activation_type'] = 'tanh' # activation function in the layers 'tanh' or None or 'relu' or 'elu'
options_vae['learning_rate'] = 0.01 # what is learning rate (usually 0.001 to 0.01)
options_vae['num_iterations'] = 5000 # number of iterations
options_vae['num_batch'] = 10 # number of batches in trainig
options_vae['print_info_step'] = 1# how many steps to show info
options_vae['sw_plot'] = 1 # switch for plotting the results within trials
options_vae['optimizer_type'] = 'Adam' # 'Adam' or 'GD'
options_vae['train_restore'] = 'train'


vae_model = simple_autoencoder_model(options_vae)
vae_model.build_computation_graph()
vae_model.train_model()

reconstruct_val,latents_val = vae_model.sess.run([vae_model.target_pred_mu,vae_model.latent_mu], feed_dict = {vae_model.X: x_train})

# plot
fig2 = plt.figure(figsize=[12,9])
ax2 = fig2.add_subplot(111,projection='3d')

ax2.scatter3D(reconstruct_val[:,0],reconstruct_val[:,1],reconstruct_val[:,2],'k.')
ax2.set_xlabel('$reconstf_1$')
ax2.set_ylabel('$recons_tf_2$')
ax2.set_zlabel('$recons_tf_1$') 
ax2.set_title('reconstructed manifold with AE')
plt.pause(0.05)
plt.show()
if options_vae['dim_hidden'] == 2:
    fig5 = plt.figure(figsize=[12,9])
    ax5 = fig5.add_subplot(111)
    ax5.plot(latents_val[:,0],latents_val[:,1],'r.')
    ax5.set_xlabel('$latent_tf_1$')
    ax5.set_ylabel('$latent_tf_2$')
    ax5.set_title('latent (low-dim) codes from AE')
    plt.pause(0.05)
    plt.show()
elif options_vae['dim_hidden'] == 1:
    fig5 = plt.figure(figsize=[12,9])
    ax5 = fig5.add_subplot(111)
    ax5.plot( range(latents_val.shape[0]) , latents_val , 'r.')

    ax5.set_xlabel('$latent_tf_1$')
    ax5.set_title('latent (low-dim) codes from AE')
    plt.pause(0.05)
    plt.show()



from sklearn.decomposition import PCA
num_pc_components = 2
pca = PCA(n_components=num_pc_components)
x_train_pc = pca.fit_transform(x_train)
reconstruct_pc_val = pca.inverse_transform(x_train_pc)
fig3 = plt.figure(figsize=[12,9])
ax3 = fig3.add_subplot(111)

ax3.plot(x_train_pc[:,0],x_train_pc[:,1],'r.')
ax3.set_xlabel('$pc_1$')
ax3.set_ylabel('$pc_2$')
ax3.set_title('principal components from PCA')
plt.pause(0.05)
plt.show()

fig4 = plt.figure(figsize=[12,9])
ax4 = fig4.add_subplot(111,projection='3d')

ax4.scatter3D(reconstruct_pc_val[:,0],reconstruct_pc_val[:,1],reconstruct_pc_val[:,2],'g.')
ax4.set_xlabel('$recons_1$')
ax4.set_ylabel('$recons_2$')
ax4.set_ylabel('$recons_3$')
ax4.set_title('reconstructed manifold from PCA ({} PCs)'.format(num_pc_components))
plt.pause(0.05)
plt.show()