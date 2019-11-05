# dataset method to handle dataset manipulation
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import stats, io

class data_set_model(object):
    # this class creates the RNN model for movement generation
    
    def __init__(self,x_train,x_test,options):
        # options: options of loading the RNN model
        # 
        self.x_train = x_train
        self.x_test = x_test
        self.dim_x = np.shape(x_test)[1]

