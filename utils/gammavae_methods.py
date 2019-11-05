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

class gvae_model(object):
    # this class creates the RNN model for movement generation
    
    def __init__(self,data_set,gamma,capacity_schedule,options):
        # options: options of loading the RNN model
        # 
        self.options = options
        self.dim_latent = options['dim_latent'] # dimension of latent variabels
        self.activation_type = options['activation_type'] # activation function in the layers 'tanh' or None or 'relu' or 'elu'
        self.learning_rate = options['learning_rate'] # what is learning rate (usually 0.001 to 0.01)
        #self.num_iterations_per_epoch = options['num_iterations_per_epoch'] # number of iterations per epoch
        self.num_epochs = options['num_epochs'] # number of epochs
        self.batch_size = options['batch_size'] # number of batches in trainig
        self.print_info_step = options['print_info_step'] # how many steps to show info
        self.sw_plot = options['sw_plot'] # switch for plotting the results within trials
        self.plot_mode = options['plot_mode'] # 'show' to show the plot or 'save' to only save the plot
        self.optimizer_type = options['optimizer_type'] # 'Adam' or 'GD'
        self.train_restore = options['train_restore'] # whether we learn or restore (maybe delete later)
        self.iteration_plot_show = options['iteration_plot_show']
        self.dropout_rate = options['dropout_rate']

        # create train and test dataset
        self.x_train = data_set.train_images
        self.x_train_num_samples = np.shape(self.x_train)[0]

        self.x_test_recons = data_set.test_reconstruct
        self.x_test_recons_num_samples = np.shape(self.x_test_recons)[0]

        self.x_test_interp = data_set.test_interpolate
        self.x_test_interp_num_samples = np.shape(self.x_test_interp)[0]
        
        #set additional parameters
        self.num_iterations_per_epoch = np.int(np.floor(self.x_train_num_samples/self.batch_size))
        self.gamma = gamma
        self.capacity_schedule = capacity_schedule
        # self.save_directory = options['save_directory'] # directory to save the results
        # self.save_folder_name = options['save_folder_name'] # file name to save the results   
        self.save_main_folder = 'gamma_vae/trained_model_gamma{:.1f}_cschedule_{}to{}'.format(self.gamma, int( np.min(self.capacity_schedule) ), int( np.max(self.capacity_schedule) ) ) 
    def recognition_network(self,recog_input):
        # we want to build this recognition network:
        # 64x64x1
        # 32x32x32
        # 16x16x32
        # 8x8x32
        # 4x4x32
        # 256
        # 256
        # 10(mu) and 10(log_sigma2)
        
        initilizer_conv = tf.random_normal_initializer(0,0.03)
        initilizer_dense = tf.contrib.layers.xavier_initializer(uniform=False)


        net = recog_input
        net = tf.expand_dims(net,axis=3)
        # conv nets
        net = tf.keras.layers.Conv2D(32,(4,4), strides = (2,2), padding = 'same', kernel_initializer = initilizer_conv) (net) #32x32x32
        net = tf.nn.relu(net)
        #net = tf.layers.dropout(net,rate = self.dropout_rate , training= self.is_training )

        net = tf.keras.layers.Conv2D(32,(4,4), strides = (2,2), padding = 'same', kernel_initializer = initilizer_conv) (net)#16x16x32
        net = tf.nn.relu(net)
        #net = tf.layers.dropout(net,rate = self.dropout_rate , training= self.is_training )

        net = tf.keras.layers.Conv2D(32,(4,4), strides = (2,2), padding = 'same', kernel_initializer = initilizer_conv) (net)#8x8x32
        net = tf.nn.relu(net)
        #net = tf.layers.dropout(net,rate = self.dropout_rate , training= self.is_training )

        net = tf.keras.layers.Conv2D(32,(4,4), strides = (2,2), padding = 'same', kernel_initializer = initilizer_conv) (net)#4x4x32
        net = tf.nn.relu(net)
        # dense layer
        net = tf.reshape(net, [-1, 4*4*32])

        net = tf.layers.dense(net,256, kernel_initializer = initilizer_dense)
        net = tf.nn.relu(net)

        net = tf.layers.dense(net,256, kernel_initializer = initilizer_dense)
        net = tf.nn.relu(net)

        mu = tf.layers.dense(net,self.dim_latent, kernel_initializer = initilizer_dense)


        # sigma2  = tf.layers.dense(net,self.dim_latent, kernel_initializer = initilizer_dense)
        # sigma2 = tf.math.square(sigma2)
        # log_sigma2 = tf.math.log(sigma2+1e-8)
        log_sigma2 = tf.layers.dense(net,self.dim_latent, kernel_initializer = initilizer_dense)
        

        return mu, log_sigma2

    def generative_network(self,latent_sample):
        # we want to build this recognition network:
        # 10
        # 256
        # 256
        # 4x4x32
        # 8x8x32
        # 16x16x32
        # 32x32x32
        # 64x64x1
        
        initilizer_dconv = tf.random_normal_initializer(0,0.03)
        initilizer_dense = tf.contrib.layers.xavier_initializer(uniform=False)
        
        net = latent_sample

        net = tf.layers.dense(net,256, kernel_initializer = initilizer_dense)
        net = tf.nn.relu(net)

        net = tf.layers.dense(net,256, kernel_initializer = initilizer_dense)
        net = tf.nn.relu(net)
    
        net = tf.layers.dense(net,512, kernel_initializer = initilizer_dense)
        net = tf.nn.relu(net)

        net = tf.reshape(net, [-1, 4, 4, 32])

        net = tf.keras.layers.Conv2DTranspose(32,kernel_size = 2, strides = 2, padding = 'valid', output_padding = 0, kernel_initializer = initilizer_dconv)(net) #8x8x32
        net = tf.nn.relu(net)
        #net = tf.layers.dropout(net,rate = self.dropout_rate , training= self.is_training )

        net = tf.keras.layers.Conv2DTranspose(32,kernel_size = 2, strides = 2, padding = 'valid', output_padding = 0, kernel_initializer = initilizer_dconv)(net) #16x16x32
        net = tf.nn.relu(net)
        #net = tf.layers.dropout(net,rate = self.dropout_rate , training= self.is_training )

        net = tf.keras.layers.Conv2DTranspose(32,kernel_size = 2, strides = 2, padding = 'valid', output_padding = 0, kernel_initializer = initilizer_dconv)(net) #32x32x32
        net = tf.nn.relu(net)
        #net = tf.layers.dropout(net,rate = self.dropout_rate , training= self.is_training )

        net = tf.keras.layers.Conv2DTranspose(1,kernel_size = 2, strides = 2, padding = 'valid', output_padding = 0, kernel_initializer = initilizer_dconv)(net) #64x64x1
        #net = tf.nn.relu(net)
        #net = tf.layers.dropout(net,rate = self.dropout_rate , training= self.is_training )


        x_reconstruct_logit = tf.squeeze(net,axis=3)
        

        return x_reconstruct_logit

    def build_computation_graph(self):
        self.sess = tf.Session()
        which_initlizer = tf.contrib.layers.xavier_initializer(uniform=False) # or None
        # define auto encoder architecture
        self.x = tf.placeholder(tf.float32 , shape=[None,64,64] ) # num_batches, smaples in each batch, target_dim
        self.lr = tf.placeholder(tf.float32 , shape=[] )
        self.is_training = tf.placeholder(tf.bool, [])
        self.c = tf.placeholder(tf.float32,[])

        self.latent_mu, self.latent_log_sigma2 = self.recognition_network(self.x)

        eps = tf.random_normal(tf.shape(self.latent_log_sigma2), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon')
                       
        self.latent_sample = self.latent_mu + tf.exp(self.latent_log_sigma2/2) * eps

        self.x_recons_logit = self.generative_network(self.latent_sample)
        self.x_recons = tf.nn.sigmoid(self.x_recons_logit)

        # define loss function
        self.loss_train = self.compute_loss()
        

        # define optimizer
        if self.optimizer_type == 'Adam':
            self.optimizer_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate)    
        elif self.optimizer_type == 'GD':
            self.optimizer_train = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)   

        # define operation of training
        self.operation_train_early = self.optimizer_train.minimize(self.recons_loss)
        self.operation_train = self.optimizer_train.minimize(self.loss_train)

        # initlize the training variables values
        self.init = tf.global_variables_initializer() 
        # saver for later work
        self.saver = tf.train.Saver()

        return       
    def gauss_sample(self,mu, var):
        epsilon = tf.random_normal(tf.shape(var), name="epsilon")
        return mu + tf.sqrt(var) * epsilon

    def compute_loss(self):
        # Reconstruction loss
        recons_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x,
                                                                logits=self.x_recons_logit)
        recons_loss = tf.reduce_sum(recons_loss, 1)
        self.recons_loss = tf.reduce_mean(recons_loss)

        # Latent loss
        latent_loss = -0.5 * tf.reduce_sum(1 + self.latent_log_sigma2
                                        - tf.square(self.latent_mu)
                                        - tf.exp(self.latent_log_sigma2), 1)
        self.latent_loss = tf.reduce_mean(latent_loss)        
        # Loss with encoding capacity term
        return self.recons_loss + self.gamma * tf.abs(self.latent_loss-self.c)

    def log_gaussian(self,x, mu, log_sigma2):
        const_log_pdf = (- 0.5 * np.log(2 * np.pi)).astype('float32') 
        #return const_log_pdf - tf.log(var) / 2 - tf.square(x - mu) / (2 * var)
        return const_log_pdf - log_sigma2 / 2 - tf.square((x - mu)) / (2 * tf.exp(log_sigma2))
     
    def train_model (self):
        # iterations to plot

        # create the loss_list
        self.loss_tot_train = []
        self.loss_recons_train = []
        self.loss_latent_train = []
        self.loss_tot_test = []
        self.loss_recons_test = []
        self.loss_latent_test = []
        #self.loss_val_iter = []


        if self.train_restore == 'train':        
            # initilize
            self.sess.run(self.init)
            for epoch in range(1,self.num_epochs+1):
                c_this = self.capacity_schedule[epoch-1]
                shuffle_index = np.random.randint(self.x_train_num_samples, size=(self.x_train_num_samples))
        
                for iteration in range(1,self.num_iterations_per_epoch+1):
                    batch_index = shuffle_index[ 
                        range( (iteration-1)*self.batch_size , (iteration)*self.batch_size) ]
                    # create the batches of input and target TODO: make it batch based
                    batch_x = self.x_train[batch_index,:,:]
                    # run gradient descent
                    if epoch < 10:
                        self.sess.run(self.operation_train_early, feed_dict = { self.x: batch_x,
                        self.c: c_this, self.is_training: True} )
                    if epoch >= 10:
                        self.sess.run(self.operation_train, feed_dict = { self.x: batch_x,
                        self.c: c_this, self.is_training: True} )   
                    # plot
                    if self.sw_plot == 1:
                        plot_folder = '{}/plots'.format(self.save_main_folder)
                        plot_name = 'epoch_{}_train.png'.format(epoch)
                        self.plot_train(iteration, self.iteration_plot_show, 5, batch_x, self.plot_mode, plot_folder, plot_name)                   
                    # display training results
                    if iteration % self.print_info_step == 0:
                        loss_this, recons_loss_this, latent_loss_this= self.sess.run([self.loss_train,self.recons_loss,self.latent_loss], feed_dict = {self.x: batch_x, self.c: c_this, self.is_training: True})
                        # print info                        
                        print('epoch {}, iteration {} --> loss_tot = {}, loss_rec = {}, loss_lat = {}'.format(epoch,iteration,loss_this, recons_loss_this, latent_loss_this))    
                        # append loss
                        self.loss_tot_train.append(loss_this)
                        self.loss_recons_train.append(recons_loss_this)
                        self.loss_latent_train.append(latent_loss_this)
                
                # create the batches of input and target TODO: make it batch based
                batch_x_test = self.x_test_recons[:,:,:]
                # run gradient descent
                loss_this, recons_loss_this, latent_loss_this = self.sess.run([self.loss_train,self.recons_loss,self.latent_loss], feed_dict = {self.x: batch_x_test,
                self.c:c_this, self.is_training: False})
                print('TEST: epoch {}--> loss_tot = {}, loss_rec = {}, loss_lat = {}'.format(epoch,loss_this, recons_loss_this, latent_loss_this))
                self.loss_tot_test.append(loss_this)
                self.loss_recons_test.append(recons_loss_this)
                self.loss_latent_test.append(latent_loss_this)
                print( 'loss_test so far: {}'.format(self.loss_tot_test) )
                if self.sw_plot == 1:
                        plot_folder = '{}/plots'.format(self.save_main_folder)
                        plot_name = 'epoch_{}_test.png'.format(epoch)
                        self.plot_train(1, 1, 8, batch_x_test, self.plot_mode, plot_folder, plot_name) 
                # save the model
                self.save_model(epoch)
                
        return 

    def save_model(self,epoch_num):
        save_folder = "{}/epoch_{}".format(self.save_main_folder, epoch_num)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print('This directory was created: {}'.format(save_folder))

        file_path = "{}/model.ckpt".format(save_folder)
        print('model is being saved as {}'.format(file_path))
        save_path = self.saver.save(self.sess, file_path)

        return

    def load_model(self, epoch_num):
        model_path = "{}/epoch_{}/model.ckpt".format(self.save_main_folder, epoch_num)
        self.build_computation_graph()
        self.saver.restore(self.sess,model_path)

    def plot_train(self,iteration,iteration_plot,num_images,batch_x,plot_mode,plot_folder,plot_name):
        # plot_folder: path to save the plot
        # plot_name: name of the save plot
        # plot_mode: 'save' or 'show'

        if  iteration % iteration_plot == 0:
            # plot prediction
                
                x_recons, z_sample, x_true = self.sess.run([self.x_recons,self.latent_sample,self.x],feed_dict = {self.x: batch_x, self.is_training: True})
                
                x_recons_binary = x_recons >= 0.5
                fig = plt.figure(figsize=[12,9])
                for img in range(num_images):
                    # ax = fig.add_subplot(100*num_images + 10*3 + 3*img + 1)
                    # ax.imshow(np.squeeze(x_true[img,:,:]) )
                    # ax.set_title('ground truth(0)')
                    # ax = fig.add_subplot(100*num_images + 10*3 + 3*img + 2)
                    # ax.imshow(np.squeeze(x_recons[img,:,:]))
                    # ax.set_title('sigmoid 0')
                    # ax = fig.add_subplot(100*num_images + 10*3 + 3*img + 3)
                    # ax.imshow(np.squeeze(x_recons_binary[img,:,:]) )
                    # ax.set_title('reconstruction 0')
                    ax = plt.subplot2grid((num_images,3),(img,0))
                    ax.imshow(np.squeeze(x_true[img,:,:]) )
                    ax.set_title('ground truth(0)')
                    ax = plt.subplot2grid((num_images,3),(img,1))
                    ax.imshow(np.squeeze(x_recons[img,:,:]))
                    ax.set_title('sigmoid 0')
                    ax = plt.subplot2grid((num_images,3),(img,2))
                    ax.imshow(np.squeeze(x_recons_binary[img,:,:]) )
                    ax.set_title('reconstruction 0')
                if plot_mode is 'show':
                    plt.pause(0.05)
                    plt.show() 
                if plot_mode is 'save': 
                    if not os.path.exists(plot_folder):
                        os.makedirs(plot_folder)
                        print('This directory was created: {}'.format(plot_folder))

                    plt.savefig('{}/{}'.format(plot_folder,plot_name))

            
            
        pass

    def reconstruct(self,x):
        x_recons = self.sess.run(self.x_recons,feed_dict = {self.x: x, self.is_training: False})
        return x_recons

    def project_to_latent(self,x):
        z_mean, z_log_sigma2 = self.sess.run([self.latent_mu, self.latent_log_sigma2],feed_dict = {self.x: x, self.is_training: False})
        return np.asarray(z_mean,dtype='float32'), np.asarray(z_log_sigma2,dtype='float32')

    def reconstruct_from_latent(self,z):
        x_recons = self.sess.run(self.x_recons,feed_dict = {self.latent_sample: z, self.is_training: False})
        return x_recons    

    def plot_latent_interpolate(self,batch_x_a,batch_x_b):
        num_images = np.shape(batch_x_a)[0]
        num_interp_value = 6        
        plot_mode = 'show'
        z_a,z_a_var = self.project_to_latent(batch_x_a)
        z_b,z_a_var = self.project_to_latent(batch_x_b)        
        
        alpha_interp = np.linspace(0,1,num_interp_value )
        interp_images = []
        interp_images.append(batch_x_a)
        for alpha in alpha_interp:
            z_this = alpha * z_b + (1-alpha) * z_a
            x_recons_this = self.reconstruct_from_latent(z_this)
            x_recons_this = x_recons_this >= 0.5
            interp_images.append(x_recons_this)
        interp_images.append(batch_x_b)


        fig = plt.figure(figsize=[15,9])
        for img in range(num_images):
            for interp_value in range(num_interp_value+2):
            # ax = plt.subplot2grid((num_images,num_interp_value),(img,interp_value))
            # ax.imshow(np.squeeze(x_true[img,:,:]) )
            # ax.set_title('ground truth(0)')
            # ax = plt.subplot2grid((num_images,num_interp_value),(img,interp_value))
            # ax.imshow(np.squeeze(x_recons[img,:,:]))
            # ax.set_title('sigmoid 0')
                ax = plt.subplot2grid( (num_images,num_interp_value+2) , (img,interp_value) )
                ax.imshow(np.squeeze(interp_images[interp_value][img,:,:]) )
                if interp_value is not 0 and interp_value is not num_interp_value+1:
                    ax.set_title('pair{},interp{}'.format(img,interp_value))
                elif interp_value is 0:
                    ax.set_title('pair{},first image'.format(img))
                elif interp_value is num_interp_value+1:
                    ax.set_title('pair{},second image'.format(img))
        if plot_mode is 'show':
            plt.pause(0.05)
            plt.show() 
        if plot_mode is 'save': 
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)
                print('This directory was created: {}'.format(plot_folder))

            plt.savefig('{}/{}'.format(plot_folder,plot_name))

            
            
        pass
