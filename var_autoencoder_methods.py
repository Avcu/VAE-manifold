#
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import stats, io
import tensorflow as tf
from tensorflow.contrib import rnn
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# methods for RNN

class var_autoencoder_model(object):
    # this class creates the RNN model for movement generation
    
    def __init__(self,options):
        # options: options of loading the RNN model
        # 
        self.options = options
        self.dim_target = options['dim_target'] # dimension of target signal
        self.dim_target_pre = options['dim_target_pre'] # dimension of encoder's layer before latent variabless
        self.dim_hidden = options['dim_hidden'] # dimension of latent variabels
        self.dim_hidden_pre = options['dim_hidden_pre'] # dimension of hidden layer before latent variabless
        self.target_train = options['target_train'] # dimension of hidden layer before latent variabless
        self.activation_type = options['activation_type'] # activation function in the layers 'tanh' or None or 'relu' or 'elu'
        self.learning_rate = options['learning_rate'] # what is learning rate (usually 0.001 to 0.01)
        self.num_iterations = options['num_iterations'] # number of iterations
        self.num_batch = options['num_batch'] # number of batches in trainig
        self.print_info_step = options['print_info_step'] # how many steps to show info
        self.sw_plot = options['sw_plot'] # switch for plotting the results within trials
        self.optimizer_type = options['optimizer_type'] # 'Adam' or 'GD'
        self.train_restore = options['train_restore'] # whether we learn or restore (maybe delete later)
        self.output_generate_type = options['output_generate_type'] # 'gauss_distrib' or 'gauss_point_estimate' or 'bernoulli_distrib' 
        # self.save_directory = options['save_directory'] # directory to save the results
        # self.save_folder_name = options['save_folder_name'] # file name to save the results   
    
    def gauss_sample(self,mu, var):
        epsilon = tf.random_normal(tf.shape(var), name="epsilon")
        return mu + tf.sqrt(var) * epsilon


    def build_computation_graph(self):
        which_initlizer = tf.contrib.layers.xavier_initializer(uniform=False) # or None
        # define auto encoder architecture
        self.X = tf.placeholder(tf.float32,shape=[None,None,self.dim_target]) # num_batches, smaples in each batch, target_dim

        # create the variables needed for projection from last decoder's layer to latent variables
        # self.weights_dec_mu = tf.Variable( tf.random_normal( [self.dim_hidden_pre,self.dim_hidden] ), name = 'proj_w_dec_mu' )
        # self.bias_dec_mu = tf.Variable( tf.random_normal( [self.dim_hidden] ), name = 'proj_b_dec_mu' )
        # self.weights_dec_log_sigma2 = tf.Variable( tf.random_normal( [self.dim_hidden_pre,self.dim_hidden] ), name = 'proj_w_dec_log_sigma2' )
        # self.bias_dec_log_sigma2 = tf.Variable( tf.random_normal( [self.dim_hidden] ), name = 'proj_b_dec_log_sigma2' )
        # create the variables needed for projection from last encoder's layer to target
        # self.weights_enc_mu = tf.Variable( tf.random_normal( [self.dim_target_pre,self.dim_target] ), name = 'proj_w_enc_mu' )
        # self.bias_enc_mu = tf.Variable( tf.random_normal( [self.dim_target] ), name = 'proj_b_enc_mu' )
        # self.weights_enc_log_sigma2 = tf.Variable( tf.random_normal( [self.dim_target_pre,self.dim_target] ), name = 'proj_w_enc_log_sigma2' )
        # self.bias_enc_log_sigma2 = tf.Variable( tf.random_normal( [self.dim_target] ), name = 'proj_b_enc_log_sigma2' )
        #
        n_hidden_recog = 5
        n_hidden_gener = 5
        
        hidden_recog_1 = tf.layers.dense(self.X , n_hidden_recog , activation=self.activation_type,
        kernel_initializer = which_initlizer)
        hidden_pre = tf.layers.dense(hidden_recog_1 , self.dim_hidden_pre , activation=self.activation_type,
        kernel_initializer = which_initlizer)

        #self.latent_mu = tf.tensordot( hidden_pre , self.weights_dec_mu , axes=[[2],[0]]) + self.bias_dec_mu
        self.latent_mu = tf.layers.dense( hidden_pre , self.dim_hidden , activation = None)
        # self.latent_log_sigma2 = tf.tensordot( hidden_pre , self.weights_dec_log_sigma2 , axes=[[2],[0]]) + self.bias_dec_log_sigma2
        self.latent_log_sigma2 = tf.layers.dense( hidden_pre , self.dim_hidden , activation = None)
        # reparametrtization trick (Gaussian sampling)
        eps = tf.random_normal(tf.shape(self.latent_log_sigma2), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon')
                       
        self.latent_sample = self.latent_mu + tf.exp(self.latent_log_sigma2/2) * eps

        #Wrong: self.latent_sample = self.gauss_sample( self.latent_mu , tf.math.exp(self.latent_log_sigma2) )

        hidden_gener_1 = tf.layers.dense(self.latent_sample , n_hidden_gener , activation=self.activation_type,
        kernel_initializer = which_initlizer)
        self.target_pre = tf.layers.dense(hidden_gener_1 , self.dim_target_pre , activation=self.activation_type,
        kernel_initializer = which_initlizer)

        # self.target_pred_mu = tf.tensordot( self.target_pre , self.weights_enc_mu , axes=[[2],[0]]) + self.bias_enc_mu
        # self.target_pred_log_sigma2 = tf.tensordot( self.target_pre , self.weights_enc_log_sigma2 , axes=[[2],[0]]) + self.bias_enc_log_sigma2

        self.target_pred_mu = tf.layers.dense( self.target_pre , self.dim_target , activation = None)
        if self.output_generate_type is 'gauss_distrib':
            self.target_pred_log_sigma2 = tf.layers.dense( self.target_pre , self.dim_target , activation = None)
            
        # define loss function
        self.loss_train = self.compute_loss()
        

        # define optimizer
        if self.optimizer_type == 'Adam':
            self.optimizer_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate)    
        elif self.optimizer_type == 'GD':
            self.optimizer_train = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)   

        # define operation of training
        self.operation_train = self.optimizer_train.minimize(self.loss_train)

        # initlize the training variables values
        self.init = tf.global_variables_initializer() 
        # saver for later work
        self.saver = tf.train.Saver()

        return       

    def log_gaussian(self,x, mu, log_sigma2):
        const_log_pdf = (- 0.5 * np.log(2 * np.pi)).astype('float32') 
        #return const_log_pdf - tf.log(var) / 2 - tf.square(x - mu) / (2 * var)
        return const_log_pdf - log_sigma2 / 2 - tf.square((x - mu)) / (2 * tf.exp(log_sigma2))
        
    def compute_loss(self):
        if self.output_generate_type is 'gauss_distrib':
            return -self.compute_elbo( self.latent_mu , self.latent_log_sigma2 , self.target_pred_mu , self.target_pred_log_sigma2 , self.X)

        elif self.output_generate_type is 'gauss_point_estimate':
            prior_sigma2 = 1.0
            KL_divergence = 0.5 * (tf.log(prior_sigma2) - self.latent_log_sigma2 + tf.math.exp(self.latent_log_sigma2)/prior_sigma2  + tf.math.square(self.latent_mu)/prior_sigma2 - 1)
            KL_divergence_mean = tf.reduce_mean( KL_divergence )

            squared_error = tf.square(self.X - self.target_pred_mu)
            squared_error_mean = tf.reduce_mean(squared_error)
            return  squared_error_mean        

    def compute_elbo(self,latent_mu,latent_log_sigma2,target_pred_mu,target_pred_log_sigma2,target):
        # the prior on latent variables are N(0,I)
        prior_sigma2 = 1.0
        KL_divergence = 0.5 * (tf.log(prior_sigma2)-latent_log_sigma2 + tf.math.exp(latent_log_sigma2)/prior_sigma2  + tf.math.square(latent_mu)/prior_sigma2 - 1)
        KL_divergence_mean = tf.reduce_mean( KL_divergence )
        gauss_log_likelihood = self.log_gaussian(target,target_pred_mu,target_pred_log_sigma2)
        gauss_log_likelihood_mean = tf.reduce_mean( gauss_log_likelihood )
        return  -KL_divergence_mean + gauss_log_likelihood_mean
        # return gauss_log_likelihood_mean
        # return -KL_divergence_mean

    def train_model (self):
        # iterations to plot

        # create the loss_list
        self.loss_iter = []
        #self.loss_val_iter = []

        # start training
        with tf.Session() as self.sess:
            if self.train_restore == 'train':        
                # initilize
                self.sess.run(self.init)

                for iteration in range(1,self.num_iterations+1):
                    # create the batches of input and target TODO: make it batch based
                    batch_x = np.expand_dims(self.target_train,axis=0)
                    # plot
                    if self.sw_plot == 1:
                        self.plot_train(iteration,500 ,batch_x)                   
                    # run gradient descent
                    self.sess.run(self.operation_train, feed_dict = { self.X: batch_x} )
                    # display training results
                    if iteration % self.print_info_step == 0:
                        loss_this= self.sess.run([self.loss_train], feed_dict = {self.X: batch_x})
                    # print info                        
                    print('iteration {} --> loss = {}'.format(iteration,loss_this))    
                    # append loss
                    self.loss_iter.append(loss_this)
                    
                # save the model
                self.save_model(self.num_iterations)
                
        return 

    def save_model(self,iteration):
        #with tf.Session() as self.sess:
        # save the sessions
        # save_folder = "{}/{}/graph_iter{}".format(self.save_directory,self.save_folder_name,iteration)
        # file_path_sess = "{}/{}/graph_iter{}/train_iter{}.ckpt".format(self.save_directory,self.save_folder_name,iteration,iteration)
        # if not os.path.exists(save_folder):
        #     os.makedirs(save_folder)
        #     print('This directory was created: {}'.format(save_folder))
        # if not os.path.isfile(file_path_sess):
        #     save_path = self.saver.save(self.sess, "{}/{}/train_iter{}.ckpt".format(self.save_directory,self.save_folder_name,iteration))
        #     print("Model saved in file: %s" % save_path) 
        # else:
        #     print('this file already exists: {}'.format(file_path_sess))


        # # get the variables we want
        # file_path_train = "{}/{}/train_iter{}_values".format(self.save_directory,self.save_folder_name,iteration)

        # batch_x_main = np.empty(shape=[1,np.shape(self.target_main)[0],self.dim_input],dtype='float32')
        # batch_y_main = np.empty(shape=[1,np.shape(self.target_main)[0],self.dim_target],dtype='float32') 
        # batch_x_main[0,:,:] = self.input_main
        # batch_y_main[0,:,:] = self.target_main
        # ts_length_main =  np.ones(shape= [1],dtype='int32') * np.shape(self.target_main)[0]
        # states , predicted_target , loss = self.sess.run([self.states_rnn , self.predicted_target , self.loss_train], feed_dict={self.X: batch_x_main , self.Y: batch_y_main , self.time_series_length: ts_length_main}) 

        # savedict = {}
        # savedict['states'] = states[0,:,:]
        # savedict['predicted_target'] = predicted_target[0,:,:]
        # savedict['target_main'] = self.target_main
        # savedict['loss'] = loss 
        # savedict['input_main'] = self.input_main        
        # savedict['options_rnn'] = self.options
        # # for matfile save
        # io.savemat('{}.mat'.format(file_path_train) , savedict )
        # #for numpy save
        # #np.savez(file_path_train,**train_vals)
        # print('saving is done')
        pass

    def plot_train(self,iteration,iteration_plot,batch_x):
        

        if  iteration % iteration_plot == 0:
            which_batch = 0
            reconstruct_val,latents_val,latents_log_sigma2 = self.sess.run([self.target_pred_mu,self.latent_mu,self.latent_log_sigma2], feed_dict = {self.X: batch_x})
            # test_plot
            fig2 = plt.figure(figsize=[18,9])
            ax2 = fig2.add_subplot(221,projection='3d')

            ax2.scatter3D(batch_x[which_batch,:,0],batch_x[which_batch,:,1],batch_x[which_batch,:,2],'k.')
            ax2.set_xlabel('$dim_1$')
            ax2.set_ylabel('$dim_2$')
            ax2.set_zlabel('$dim_3$') 
            ax2.set_title('True manifold')

            ax2 = fig2.add_subplot(222,projection='3d')

            ax2.scatter3D(reconstruct_val[which_batch,:,0],reconstruct_val[which_batch,:,1],reconstruct_val[which_batch,:,2],'k.')
            ax2.set_xlabel('$reconstf_1$')
            ax2.set_ylabel('$recons_tf_2$')
            ax2.set_zlabel('$recons_tf_1$') 
            ax2.set_title('reconstructed manifold with AE')
            if self.dim_hidden >= 2:
                ax5 = fig2.add_subplot(223)
                ax5.plot(latents_val[which_batch,:,0],latents_val[which_batch,:,1],'r.')
                ax5.set_xlabel('$latent_tf_1$')
                ax5.set_ylabel('$latent_tf_2$')
                ax5.set_title('latent (low-dim) codes from AE')
                plt.pause(0.05)
                plt.show()
            elif self.dim_hidden == 1:
                ax5 = fig2.add_subplot(223)
                ax5.plot( range(latents_val.shape[1]) , latents_val[which_batch,:] , 'r.')

                ax5.set_xlabel('$latent_tf_1$')
                ax5.set_title('latent (low-dim) codes from AE')
                plt.pause(0.05)
                plt.show()

            
            
        pass

class simple_autoencoder_model(object):
    # this class creates the RNN model for movement generation
    
    def __init__(self,options):
        # options: options of loading the RNN model
        # 
        self.options = options
        self.dim_target = options['dim_target'] # dimension of target signal
        self.dim_target_pre = options['dim_target_pre'] # dimension of encoder's layer before latent variabless
        self.dim_hidden = options['dim_hidden'] # dimension of latent variabels
        self.dim_hidden_pre = options['dim_hidden_pre'] # dimension of hidden layer before latent variabless
        self.target_train = options['target_train'] # dimension of hidden layer before latent variabless
        self.activation_type = options['activation_type'] # activation function in the layers 'tanh' or None or 'relu' or 'elu'
        self.learning_rate = options['learning_rate'] # what is learning rate (usually 0.001 to 0.01)
        self.num_iterations = options['num_iterations'] # number of iterations
        self.num_batch = options['num_batch'] # number of batches in trainig
        self.print_info_step = options['print_info_step'] # how many steps to show info
        self.sw_plot = options['sw_plot'] # switch for plotting the results within trials
        self.optimizer_type = options['optimizer_type'] # 'Adam' or 'GD'
        self.train_restore = options['train_restore'] # whether we learn or restore (maybe delete later)
        # self.save_directory = options['save_directory'] # directory to save the results
        # self.save_folder_name = options['save_folder_name'] # file name to save the results   


    def build_computation_graph(self):
        # define auto encoder architecture
        self.X = tf.placeholder(tf.float32,shape=[None,None,self.dim_target])

        hidden_recog_1 = tf.layers.dense(self.X , self.dim_hidden_pre , activation=self.activation_type)
        hidden_recog_2 = tf.layers.dense(hidden_recog_1 , self.dim_hidden_pre , activation=self.activation_type)
        self.latent = tf.layers.dense(hidden_recog_2 , self.dim_hidden , activation=self.activation_type)
        hidden_gener_1 = tf.layers.dense(self.latent , self.dim_target_pre , activation=self.activation_type)
        hidden_gener_2 = tf.layers.dense(hidden_gener_1 , self.dim_target_pre , activation=self.activation_type)
        self.target_pred = tf.layers.dense(hidden_gener_2 , self.dim_target , activation=None)

        self.loss_train = tf.reduce_mean(tf.square(self.target_pred-self.X))

        # define optimizer
        if self.optimizer_type == 'Adam':
            self.optimizer_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate)    
        elif self.optimizer_type == 'GD':
            self.optimizer_train = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)   

        # define operation of training
        self.operation_train = self.optimizer_train.minimize(self.loss_train)

        # initlize the training variables values
        self.init = tf.global_variables_initializer() 
        # saver for later work
        self.saver = tf.train.Saver()

        return       

    def train_model (self):
        # iterations to plot

        # create the loss_list
        self.loss_iter = []
        #self.loss_val_iter = []

        # start training
        with tf.Session() as self.sess:
            if self.train_restore == 'train':        
                # initilize
                self.sess.run(self.init)

                for iteration in range(1,self.num_iterations+1):
                    # create the batches of input and target TODO: make it batch based
                    batch_x = np.expand_dims(self.target_train,axis=0)
                    # plot
                    if self.sw_plot == 1:
                        self.plot_train(iteration,500 ,batch_x)
                    
                    # run gradient descent
                    self.sess.run(self.operation_train, feed_dict = { self.X: batch_x} )

                    # display training results
                    if iteration % self.print_info_step == 0:
                        loss_this= self.sess.run([self.loss_train], feed_dict = {self.X: batch_x})
                        
                        #evaluate for total evaluation time-series
                        
                    print('iteration {} --> loss = {}'.format(iteration,loss_this))    

                    self.loss_iter.append(loss_this)
                    
                # save the model
                self.save_model(self.num_iterations)
                #
                
        return 

    def save_model(self,iteration):
        #with tf.Session() as self.sess:
        # save the sessions
        # save_folder = "{}/{}/graph_iter{}".format(self.save_directory,self.save_folder_name,iteration)
        # file_path_sess = "{}/{}/graph_iter{}/train_iter{}.ckpt".format(self.save_directory,self.save_folder_name,iteration,iteration)
        # if not os.path.exists(save_folder):
        #     os.makedirs(save_folder)
        #     print('This directory was created: {}'.format(save_folder))
        # if not os.path.isfile(file_path_sess):
        #     save_path = self.saver.save(self.sess, "{}/{}/train_iter{}.ckpt".format(self.save_directory,self.save_folder_name,iteration))
        #     print("Model saved in file: %s" % save_path) 
        # else:
        #     print('this file already exists: {}'.format(file_path_sess))


        # # get the variables we want
        # file_path_train = "{}/{}/train_iter{}_values".format(self.save_directory,self.save_folder_name,iteration)

        # batch_x_main = np.empty(shape=[1,np.shape(self.target_main)[0],self.dim_input],dtype='float32')
        # batch_y_main = np.empty(shape=[1,np.shape(self.target_main)[0],self.dim_target],dtype='float32') 
        # batch_x_main[0,:,:] = self.input_main
        # batch_y_main[0,:,:] = self.target_main
        # ts_length_main =  np.ones(shape= [1],dtype='int32') * np.shape(self.target_main)[0]
        # states , predicted_target , loss = self.sess.run([self.states_rnn , self.predicted_target , self.loss_train], feed_dict={self.X: batch_x_main , self.Y: batch_y_main , self.time_series_length: ts_length_main}) 

        # savedict = {}
        # savedict['states'] = states[0,:,:]
        # savedict['predicted_target'] = predicted_target[0,:,:]
        # savedict['target_main'] = self.target_main
        # savedict['loss'] = loss 
        # savedict['input_main'] = self.input_main        
        # savedict['options_rnn'] = self.options
        # # for matfile save
        # io.savemat('{}.mat'.format(file_path_train) , savedict )
        # #for numpy save
        # #np.savez(file_path_train,**train_vals)
        # print('saving is done')
        pass

    def plot_train(self,iteration,iteration_plot,batch_x):
        

        if  iteration % iteration_plot == 0:
            which_batch = 0
            reconstruct_val,latents_val = self.sess.run([self.target_pred,self.latent], feed_dict = {self.X: batch_x})
            # test_plot
            fig2 = plt.figure(figsize=[18,9])
            ax2 = fig2.add_subplot(221,projection='3d')

            ax2.scatter3D(batch_x[which_batch,:,0],batch_x[which_batch,:,1],batch_x[which_batch,:,2],'k.')
            ax2.set_xlabel('$dim_1$')
            ax2.set_ylabel('$dim_2$')
            ax2.set_zlabel('$dim_3$') 
            ax2.set_title('True manifold')

            ax2 = fig2.add_subplot(222,projection='3d')

            ax2.scatter3D(reconstruct_val[which_batch,:,0],reconstruct_val[which_batch,:,1],reconstruct_val[which_batch,:,2],'k.')
            ax2.set_xlabel('$reconstf_1$')
            ax2.set_ylabel('$recons_tf_2$')
            ax2.set_zlabel('$recons_tf_1$') 
            ax2.set_title('reconstructed manifold with AE')
            if self.dim_hidden >= 2:
                ax5 = fig2.add_subplot(223)
                ax5.plot(latents_val[which_batch,:,0],latents_val[which_batch,:,1],'r.')
                ax5.set_xlabel('$latent_tf_1$')
                ax5.set_ylabel('$latent_tf_2$')
                ax5.set_title('latent (low-dim) codes from AE')
                plt.pause(0.05)
                plt.show()
            elif self.dim_hidden == 1:
                ax5 = fig2.add_subplot(223)
                ax5.plot( range(latents_val.shape[1]) , latents_val[which_batch,:] , 'r.')

                ax5.set_xlabel('$latent_tf_1$')
                ax5.set_title('latent (low-dim) codes from AE')
                plt.pause(0.05)
                plt.show()

            
            
        pass    