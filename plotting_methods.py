#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#
def plot_manifold(points_manifold,options):
    dim_points = np.shape(points_manifold)[1]

    if dim_points>2:
        fig = plt.figure(figsize=[12,9])
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter3D(points_manifold[:,0],points_manifold[:,1],points_manifold[:,2],s=50,marker='o',color='g',linewidth = 3,alpha=1)
        ax.set_xlabel('dim 1')
        ax.set_ylabel('dim 2')
        ax.set_zlabel('dim 3')
        ax.set_title('true data points on manifold')
        plt.pause(0.01)
        plt.show()
    elif dim_points == 2:
        fig = plt.figure(figsize=[12,9])
        ax = fig.add_subplot(111)
        ax.scatter(points_manifold[:,0],points_manifold[:,1],s=50,marker='o',color='g',linewidth = 3,alpha=1)
        ax.set_xlabel('dim 1')
        ax.set_ylabel('dim 2')
        ax.set_title('true data points on manifold')
        plt.pause(0.01)
        plt.show() 