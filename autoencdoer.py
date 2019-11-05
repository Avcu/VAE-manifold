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
# change seed
np.random.seed(42)
m = 5000
w1, w2 = 0.1, 0.3
noise = 0.05
manifold_shape = 'linear' # 'linear' or 'nonlinear'
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

#
n_inputs = 3
n_latent = 1
n_hidden_recog = 5
n_hidden_gener = 5
n_outputs = 3
learning_rate = 0.01
activation_type = None #'tanh' or None

# define auto encoder architecture
X = tf.placeholder(tf.float32,shape=[None,n_inputs])

hidden_recog_1 = tf.layers.dense(X , n_hidden_recog , activation=activation_type)
hidden_recog_2 = tf.layers.dense(hidden_recog_1 , n_hidden_recog , activation=activation_type)
latent = tf.layers.dense(hidden_recog_2 , n_latent , activation=activation_type)
hidden_gener_1 = tf.layers.dense(latent , n_hidden_gener , activation=activation_type)
hidden_gener_2 = tf.layers.dense(hidden_gener_1 , n_hidden_gener , activation=activation_type)
outputs = tf.layers.dense(hidden_gener_2 , n_outputs , activation=None)

loss = tf.reduce_mean(tf.square(outputs-X))
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_iterations = 3000
latents = latent
output_layer = outputs
# train
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        training_op.run(feed_dict={X:x_train})
        loss_this = sess.run(loss,feed_dict={X:x_train})
        print('iteration {} ----> loss = {:.3f}'.format(iteration,loss_this))

    latents_val = latents.eval(feed_dict={X:x_train})
    reconstruct_val = output_layer.eval(feed_dict={X:x_train})
    latents_val_test = latents.eval(feed_dict={X:x_test})
    reconstruct_val_test = output_layer.eval(feed_dict={X:x_test})

# plot
fig2 = plt.figure(figsize=[12,9])
ax2 = fig2.add_subplot(111,projection='3d')

ax2.scatter3D(reconstruct_val[:,0],reconstruct_val[:,1],reconstruct_val[:,2],'k.')
ax2.scatter3D(reconstruct_val_test[:,0],reconstruct_val_test[:,1],reconstruct_val_test[:,2],s=200,marker='o',color='g',linewidth = 3)
ax2.scatter3D(x_test[:,0],x_test[:,1],x_test[:,2],s=200,marker='x',color='g',linewidth = 3)
ax2.set_xlabel('$reconstf_1$')
ax2.set_ylabel('$recons_tf_2$')
ax2.set_zlabel('$recons_tf_1$') 
ax2.set_title('reconstructed manifold with AE')
plt.pause(0.05)
plt.show()
if n_latent == 2:
    fig5 = plt.figure(figsize=[12,9])
    ax5 = fig5.add_subplot(111)
    ax5.plot(latents_val[:,0],latents_val[:,1],'r.')
    ax5.plot(latents_val_test[:,0],latents_val_test[:,1],color='g',linewidth=3)
    ax5.set_xlabel('$latent_tf_1$')
    ax5.set_ylabel('$latent_tf_2$')
    ax5.set_title('latent (low-dim) codes from AE')
    plt.pause(0.05)
    plt.show()
elif n_latent == 1:
    fig5 = plt.figure(figsize=[12,9])
    ax5 = fig5.add_subplot(111)
    ax5.plot( range(latents_val.shape[0]) , latents_val , 'r.')
    ax5.plot( range(latents_val_test.shape[0]) , latents_val_test , color='g' , linewidth=3)

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