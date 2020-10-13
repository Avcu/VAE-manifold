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
        self.noise_value = options['noise_value']
        if 'fixed_values' in options:
            self.fixed_values = options['fixed_values']
        
        self.options = options
        return
    def generate_manifold(self):

        if self.manifold_type == 'circle':
            points_latent = np.linspace(0, 2*np.pi, num = self.num_points)
            points_manifold = np.empty( ( np.shape(points_latent)[0] ,  2) , dtype='float32')
            points_manifold[:,0] = np.cos(points_latent) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,1] = np.sin(points_latent) + self.noise_value * np.random.randn(self.num_points)
            points_colors = cm.rainbow( np.linspace(0, 1, self.num_points ) )

        elif self.manifold_type == 'distorted_circle':
            points_latent = np.linspace(0, 2*np.pi, num = self.num_points)
            points_manifold = np.empty( ( np.shape(points_latent)[0] ,  3) , dtype='float32')
            points_manifold[:,0] = np.cos(points_latent) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,1] = np.sin(points_latent) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,2] = np.sin(2*points_latent) + self.noise_value * np.random.randn(self.num_points)
            points_colors = cm.rainbow( np.linspace(0, 1, self.num_points ) )
        
        elif self.manifold_type == 'sphere':
            points_latent = np.empty( ( self.num_points ,  2) , dtype='float32')
            points_latent[:,0] = np. squeeze( np.matlib.repmat ( np.expand_dims( np.linspace(0, 2*np.pi, num = self.num_points/100), axis = 1) , 100, 1 ) )
            points_latent[:,1] =  np.squeeze( np.pi * np.random.random_sample((self.num_points, 1))  )
            points_manifold = np.empty( ( np.shape(points_latent)[0] ,  3) , dtype='float32')
            points_manifold[:,0] = np.cos(points_latent[:,0]) * np.sin(points_latent[:,1]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,1] = np.sin(points_latent[:,0]) * np.sin(points_latent[:,1]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,2] = np.cos(points_latent[:,1]) + self.noise_value * np.random.randn(self.num_points)
            points_colors = np.matlib.repmat ( cm.rainbow( np.linspace(0, 1, self.num_points/100 ) ), 100, 1) 
        
        elif self.manifold_type == 'torus':
            # (R+rcos(theta)) * cos(phi)
            # (R+rcos(theta)) * sin(phi)
            # (rsin(theta)) 
            R = 8
            r = 3
            points_latent = np.empty( ( self.num_points ,  2) , dtype='float32')
            points_latent[:,0] = np. squeeze( np.matlib.repmat ( np.expand_dims( np.linspace(0, 2*np.pi, num = self.num_points/100), axis = 1) , 100, 1 ) )
            points_latent[:,1] =  np.squeeze( 2 * np.pi * np.random.random_sample((self.num_points, 1))  )
            points_manifold = np.empty( ( np.shape(points_latent)[0] ,  3) , dtype='float32')
            points_manifold[:,0] = (R + r * np.cos(points_latent[:,0])) * np.cos(points_latent[:,1]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,1] = (R + r * np.cos(points_latent[:,0])) * np.sin(points_latent[:,1]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,2] = r * np.sin(points_latent[:,0]) + self.noise_value * np.random.randn(self.num_points)
            points_colors = np.matlib.repmat ( cm.rainbow( np.linspace(0, 1, self.num_points/100 ) ), 100, 1) 

        elif self.manifold_type == 'sphere_multiple_theta':
            points_latent = np.empty( ( self.num_points ,  2) , dtype='float32')

            angle_tuple = ()
            num_fixed_values = len(self.fixed_values)
            for angle_this in self.fixed_values:
                angle_tuple = angle_tuple + (angle_this * np.ones( ( int(self.num_points/num_fixed_values) ), dtype = 'float32'),)

            points_latent[:,0] = np.concatenate( angle_tuple , axis = 0)
            points_latent[:,1] =  np. squeeze( np.matlib.repmat ( np.expand_dims( np.linspace(0, np.pi, num = self.num_points/num_fixed_values), axis = 1) , num_fixed_values, 1 ) )
            points_manifold = np.empty( ( np.shape(points_latent)[0] ,  3) , dtype='float32')
            points_manifold[:,0] = np.cos(points_latent[:,0]) * np.sin(points_latent[:,1]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,1] = np.sin(points_latent[:,0]) * np.sin(points_latent[:,1]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,2] = np.cos(points_latent[:,1]) + self.noise_value * np.random.randn(self.num_points)
            points_colors = np.matlib.repmat ( cm.rainbow( np.linspace(0, 1, self.num_points/num_fixed_values ) ), num_fixed_values, 1)  

        elif self.manifold_type == 'sphere_multiple_phi':
            points_latent = np.empty( ( self.num_points ,  2) , dtype='float32')
            z_tuple = ()
            num_fixed_values = len(self.fixed_values)
            for z_this in self.fixed_values:
                z_tuple = z_tuple + (z_this * np.ones( ( int(self.num_points/num_fixed_values) ), dtype = 'float32'),)

            points_latent[:,0] =  np. squeeze( np.matlib.repmat ( np.expand_dims( np.linspace(0, 2 * np.pi, num = self.num_points/num_fixed_values), axis = 1) , num_fixed_values, 1 ) )
            points_latent[:,1] =  np.concatenate( z_tuple , axis = 0)

            points_manifold = np.empty( ( np.shape(points_latent)[0] ,  3) , dtype='float32')
            points_manifold[:,0] = np.cos( points_latent[:,0] ) * np.sin( points_latent[:,1] ) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,1] = np.sin( points_latent[:,0] ) * np.sin( points_latent[:,1] ) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,2] = np.cos( points_latent[:,1] ) + self.noise_value * np.random.randn(self.num_points)
            points_colors = np.matlib.repmat ( cm.rainbow( np.linspace(0, 1, self.num_points/num_fixed_values ) ), num_fixed_values, 1) 

        elif self.manifold_type == 'cylinder':
            points_latent = np.empty( ( self.num_points ,  2) , dtype='float32')
            points_latent[:,0] = np. squeeze( np.matlib.repmat ( np.expand_dims( np.linspace(0, 2*np.pi, num = self.num_points/100), axis = 1) , 100, 1 ) )
            points_latent[:,1] =  np.squeeze( np.random.random_sample((self.num_points, 1))  )
            points_manifold = np.empty( ( np.shape(points_latent)[0] ,  3) , dtype='float32')
            points_manifold[:,0] = np.cos(points_latent[:,0]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,1] = np.sin(points_latent[:,0]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,2] = points_latent[:,1] + self.noise_value * np.random.randn(self.num_points)
            points_colors = np.matlib.repmat ( cm.rainbow( np.linspace(0, 1, self.num_points/100 ) ), 100, 1) 

   
        elif self.manifold_type == '3_4_cylinder':
            points_latent = np.empty( ( self.num_points ,  2) , dtype='float32')
            points_latent[:,0] = np. squeeze( np.matlib.repmat ( np.expand_dims( np.linspace(0, 2*np.pi*(3/4), num = self.num_points/100), axis = 1) , 100, 1 ) )
            points_latent[:,1] =  np.squeeze( np.random.random_sample((self.num_points, 1))  )
            points_manifold = np.empty( ( np.shape(points_latent)[0] ,  3) , dtype='float32')
            points_manifold[:,0] = np.cos(points_latent[:,0]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,1] = np.sin(points_latent[:,0]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,2] = points_latent[:,1] + self.noise_value * np.random.randn(self.num_points)
            points_colors = np.matlib.repmat ( cm.rainbow( np.linspace(0, 1, self.num_points/100 ) ), 100, 1)   

        elif self.manifold_type == 'cylinder_multiple_angle':
            points_latent = np.empty( ( self.num_points ,  2) , dtype='float32')

            angle_tuple = ()
            num_fixed_values = len(self.fixed_values)
            for angle_this in self.fixed_values:
                angle_tuple = angle_tuple + (angle_this * np.ones( ( int(self.num_points/num_fixed_values) ), dtype = 'float32'),)

            points_latent[:,0] = np.concatenate( angle_tuple , axis = 0)
            points_latent[:,1] =  np. squeeze( np.matlib.repmat ( np.expand_dims( np.linspace(0, 1, num = self.num_points/num_fixed_values), axis = 1) , num_fixed_values, 1 ) )
            points_manifold = np.empty( ( np.shape(points_latent)[0] ,  3) , dtype='float32')
            points_manifold[:,0] = np.cos(points_latent[:,0]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,1] = np.sin(points_latent[:,0]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,2] = points_latent[:,1] + self.noise_value * np.random.randn(self.num_points)
            points_colors = np.matlib.repmat ( cm.rainbow( np.linspace(0, 1, self.num_points/num_fixed_values ) ), num_fixed_values, 1)

        elif self.manifold_type == 'cylinder_multiple_z':
            points_latent = np.empty( ( self.num_points ,  2) , dtype='float32')
            z_tuple = ()
            num_fixed_values = len(self.fixed_values)
            for z_this in self.fixed_values:
                z_tuple = z_tuple + (z_this * np.ones( ( int(self.num_points/num_fixed_values) ), dtype = 'float32'),)

            points_latent[:,0] =  np. squeeze( np.matlib.repmat ( np.expand_dims( np.linspace(0, 2 * np.pi, num = self.num_points/num_fixed_values), axis = 1) , num_fixed_values, 1 ) )
            points_latent[:,1] =  np.concatenate( z_tuple , axis = 0)

            points_manifold = np.empty( ( np.shape(points_latent)[0] ,  3) , dtype='float32')
            points_manifold[:,0] = np.cos(points_latent[:,0]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,1] = np.sin(points_latent[:,0]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,2] = points_latent[:,1] + self.noise_value * np.random.randn(self.num_points)
            points_colors = np.matlib.repmat ( cm.rainbow( np.linspace(0, 1, self.num_points/num_fixed_values ) ), num_fixed_values, 1)   
        elif self.manifold_type == 'distorted_cylinder':

            points_latent = np.empty( ( self.num_points ,  2) , dtype='float32')
            points_latent[:,0] = np. squeeze( np.matlib.repmat ( np.expand_dims( np.linspace(0, 2*np.pi, num = self.num_points/100), axis = 1) , 100, 1 ) )
            points_latent[:,1] =  np.squeeze( np.random.random_sample((self.num_points, 1))  )
            r = np.abs(points_latent[:,1] - 0.5) + 0.5
            points_manifold = np.empty( ( np.shape(points_latent)[0] ,  3) , dtype='float32')
            points_manifold[:,0] = r * np.cos(points_latent[:,0]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,1] = r * np.sin(points_latent[:,0]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,2] = points_latent[:,1] + self.noise_value * np.random.randn(self.num_points)
            points_colors = np.matlib.repmat ( cm.rainbow( np.linspace(0, 1, self.num_points/100 ) ), 100, 1) 
        elif self.manifold_type == 'distorted_cylinder_multiple_angle':
            points_latent = np.empty( ( self.num_points ,  2) , dtype='float32')

            angle_tuple = ()
            num_fixed_values = len(self.fixed_values)
            for angle_this in self.fixed_values:
                angle_tuple = angle_tuple + (angle_this * np.ones( ( int(self.num_points/num_fixed_values) ), dtype = 'float32'),)

            points_latent[:,0] = np.concatenate( angle_tuple , axis = 0)
            points_latent[:,1] =  np. squeeze( np.matlib.repmat ( np.expand_dims( np.linspace(0, 1, num = self.num_points/num_fixed_values), axis = 1) , num_fixed_values, 1 ) )
            r = np.abs(points_latent[:,1] - 0.5) + 0.5
            points_manifold = np.empty( ( np.shape(points_latent)[0] ,  3) , dtype='float32')
            points_manifold[:,0] = r * np.cos(points_latent[:,0]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,1] = r * np.sin(points_latent[:,0]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,2] = points_latent[:,1] + self.noise_value * np.random.randn(self.num_points)
            points_colors = np.matlib.repmat ( cm.rainbow( np.linspace(0, 1, self.num_points/num_fixed_values ) ), num_fixed_values, 1)

        elif self.manifold_type == 'distorted_cylinder_multiple_z':
            points_latent = np.empty( ( self.num_points ,  2) , dtype='float32')
            z_tuple = ()
            num_fixed_values = len(self.fixed_values)
            for z_this in self.fixed_values:
                z_tuple = z_tuple + (z_this * np.ones( ( int(self.num_points/num_fixed_values) ), dtype = 'float32'),)

            points_latent[:,0] =  np. squeeze( np.matlib.repmat ( np.expand_dims( np.linspace(0, 2 * np.pi, num = self.num_points/num_fixed_values), axis = 1) , num_fixed_values, 1 ) )
            points_latent[:,1] =  np.concatenate( z_tuple , axis = 0)
            r = np.abs(points_latent[:,1] - 0.5) + 0.5
            points_manifold = np.empty( ( np.shape(points_latent)[0] ,  3) , dtype='float32')
            points_manifold[:,0] = r * np.cos(points_latent[:,0]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,1] = r * np.sin(points_latent[:,0]) + self.noise_value * np.random.randn(self.num_points)
            points_manifold[:,2] = points_latent[:,1] + self.noise_value * np.random.randn(self.num_points)
            points_colors = np.matlib.repmat ( cm.rainbow( np.linspace(0, 1, self.num_points/num_fixed_values ) ), num_fixed_values, 1)


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
        elif dim_points == 2:
            fig = plt.figure(figsize=[12,9])
            ax = fig.add_subplot(111)
            ax.scatter(self.points_manifold[:,0],self.points_manifold[:,1],s=50,marker='o',color=self.points_colors,linewidth = 1)
            ax.set_xlabel('dim 1')
            ax.set_ylabel('dim 2')
            ax.set_title('true data points on manifold')
        if self.manifold_type == 'torus':
            ax.view_init(elev=64., azim=100.)       
        plt.pause(0.01)
        plt.show()        