# this is the manifold utils
import numpy as np
import numpy.matlib as matlib
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class manifold_static(object):

    def __init__(self,options):

        self.manifold_type = options['manifold_type']
        self.num_points = options['num_points']

        
        return

    def generate_manifold(self):

        if self.manifold_type == 'circle':
            points_latent = np.linspace(0, 2*np.pi, num = self.num_points)
            points_manifold = np.empty( ( np.shape(points_latent)[0] ,  2) , dtype='float32')
            points_manifold[:,0] = np.cos(points_latent)
            points_manifold[:,1] = np.sin(points_latent)
            points_colors = cm.rainbow( np.linspace(0, 1, self.num_points ) )
        if self.manifold_type == 'cylinder':
            points_latent = np.empty( ( self.num_points ,  2) , dtype='float32')
            points_latent[:,0] = np. squeeze( np.matlib.repmat ( np.expand_dims( np.linspace(0, 2*np.pi, num = self.num_points/100), axis = 1) , 100, 1 ) )
            points_latent[:,1] =  np.squeeze( np.random.random_sample((self.num_points, 1))  )
            points_manifold = np.empty( ( np.shape(points_latent)[0] ,  3) , dtype='float32')
            points_manifold[:,0] = np.cos(points_latent[:,0])
            points_manifold[:,1] = np.sin(points_latent[:,0])
            points_manifold[:,2] = points_latent[:,1]
            points_colors = np.matlib.repmat ( cm.rainbow( np.linspace(0, 1, self.num_points/100 ) ), 100, 1)    


        self.points_latent = points_latent
        self.points_manifold = points_manifold 
        self.points_colors = points_colors   

        return    

    def plot_manifold(self):
        dim_points = np.shape(self.points_manifold)[1]

        if dim_points>2:
            fig = plt.figure(figsize=[12,9])
            ax = fig.add_subplot(111,projection='3d')
            ax.scatter3D(self.points_manifold[:,0],self.points_manifold[:,1],self.points_manifold[:,2],s=50,marker='o',color=self.points_colors,linewidth = 2)
            ax.set_xlabel('dim 1')
            ax.set_ylabel('dim 2')
            ax.set_zlabel('dim 3')
            ax.set_title('true data points on manifold')
            plt.pause(0.01)
            plt.show()
        elif dim_points == 2:
            fig = plt.figure(figsize=[12,9])
            ax = fig.add_subplot(111)
            ax.scatter(self.points_manifold[:,0],self.points_manifold[:,1],s=50,marker='o',color=self.points_colors,linewidth = 1)
            ax.set_xlabel('dim 1')
            ax.set_ylabel('dim 2')
            ax.set_title('true data points on manifold')
            plt.pause(0.01)
            plt.show()         