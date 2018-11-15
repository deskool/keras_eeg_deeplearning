"""
Created on Fri Jan 19 11:30:54 2018

@author: Mohammad Ghassemi, PhD Candidate, MIT
"""

###############################################################################
#IMPORT THE REQUISITE LIBRARIES ###############################################
###############################################################################
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D, Conv2DTranspose, UpSampling3D, Conv3DTranspose
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
import random, pylab, sys, keras
import keras.backend as K
import numpy as np
import tensorflow as tf  
from six.moves import reload_module
import matplotlib.pyplot as plt
import sys
# Custom Functions ------------------------------------------------------------
import custom                                                                  # Import our custom functions that make this file clean.
reload_module(custom)                                                          # Reload the custom modules (good if you're re-running)

if K.tensorflow_backend._get_available_gpus() == []:                           # Let the user know if they are not using a GPU
    print('YOU ARE NOT USING A GPU ON THIS DEVICE!') 


###############################################################################
# LOAD HYPER-PARAMETER FILES ##################################################
###############################################################################
testing           = 0
h                 = int(sys.argv[1])                                                          # Which experiment do you want to run (entry in the hyperparameters file)
hp                = np.load('hyper_parameters.npy')                            # load the hyper-parameter files 
hp                = hp.item()                                                  # unpack hyper-paramters

# -----------------------------------------------------------------------------
# Choose the neural network
# -----------------------------------------------------------------------------
#A 3D CNN      
if hp['use_CNN_3D'][h] == 1:
    nn = custom.CNN_3D(hp,h)

#A LRCN
if hp['use_LRCN'][h] == 1:
    nn = custom.LRCN(hp,h)


#A 2D CNN    
if hp['use_CNN_2D'][h] == 1:
    nn = custom.CNN_2D(hp,h) 

nn.summary()
#------------------------------------------------------------------------------
# Generate Directories for the cross-fold validation results   
#------------------------------------------------------------------------------             
custom.gen_folders(hp,h,nn) 

###############################################################################
# LOAD THE TRAINING AND TESTING DATA 
###############################################################################
np.random.seed(hp['seed'][h])

# load in the EEG data...
covars    = np.load('python_covariates/covars.npy')
header    = np.load('python_covariates/covars_headers.npy')
files     = np.load('python_covariates/python_file_locs.npy')
inst      = np.load('python_covariates/inst.npy')
outcomes  = covars[:,-1:]

# Extraxt data characteristics - each subject has been made in 600 examples...
eg_data = np.load(files[0].decode('utf8'))
n_patients , n_samples , n_frames , row , col , n_hertz = np.size(files,0) , np.size(eg_data,0) , np.size(eg_data,1) , np.size(eg_data,2) , np.size(eg_data,3) , np.size(eg_data,4)
del eg_data

# Get the list of unique institutions
u_inst = np.unique(inst)

#let's try leave-one-inst-out:
for fold_num in range(0,np.size(u_inst)):
    
    if testing == 0:
        #find the index of all good and bad outcomes
        bad_out_ind  = pylab.find(outcomes == 1)
        good_out_ind = pylab.find(outcomes == 0)
        
        # Get all subjects from a given 'inst' and make them the 'test' set.
        test_ind       = pylab.find(inst == u_inst[fold_num])
        hp['test_perc'][h]       = 1.0*np.size(test_ind)/np.size(outcomes) 
        X_test,Y_test  = custom.gen_data(test_ind,files,covars[test_ind,-1],hp['take_only'][h] )
        
        # Remove the test subjects from the  from the trianing/test set.
        bad_out_ind  =  np.setdiff1d(bad_out_ind,test_ind)
        good_out_ind = np.setdiff1d(good_out_ind,test_ind)
        
        # Take a random 50% of the remaining subjects and split them into training and validation sets.
        train_bad_out_ind    = np.random.choice(bad_out_ind, round(np.size(bad_out_ind)*(hp['train_perc'][h] /(hp['val_perc'][h] + hp['train_perc'][h] ))),replace=False)
        train_good_out_ind   = np.random.choice(good_out_ind, round(np.size(good_out_ind)*(hp['train_perc'][h] /(hp['val_perc'][h] + hp['train_perc'][h] ))),replace=False)
        train_ind            = np.append(train_bad_out_ind,train_good_out_ind)
        X_train,Y_train      = custom.gen_data(train_ind,files,covars[train_ind,-1],hp['take_only'][h] )
        
        # Leave what's left over for validation.
        val_bad_out_ind    = np.setdiff1d(bad_out_ind,train_bad_out_ind)
        val_good_out_ind   = np.setdiff1d(good_out_ind,train_good_out_ind)
        val_ind            = np.append(val_bad_out_ind,val_good_out_ind)
        X_val,Y_val        = custom.gen_data(val_ind,files,covars[val_ind,-1],hp['take_only'][h] )
    
    # If this is a 2D CNN, then you need to transform the data before continuing...
    if hp['use_CNN_2D'][h] == 1:
        [X_train,Y_train, X_val,Y_val,X_test,Y_test] = custom.convert4D_to_2D(X_train,Y_train, X_val,Y_val,X_test,Y_test)
    
    
    ###############################################################################
    # Train The Neural Network          
    ###############################################################################        
    #A 3D CNN      
    np.random.seed(hp['seed'][h])
    if hp['use_CNN_3D'][h] == 1:
        nn = custom.CNN_3D(hp,h)
    
    #A LRCN
    if hp['use_LRCN'][h] == 1:
        nn = custom.LRCN(hp,h)
     
    #A 2D CNN    
    if hp['use_CNN_2D'][h] == 1:
        nn = custom.CNN_2D(hp,h) 
    
    fit_nn = custom.train_NN(hp,h,fold_num,nn,X_train,Y_train,X_val,Y_val)
    
    #Count the number of epochs that were completed before the convergence criteria
    epochs_completed = len(fit_nn.history['loss'])
     
    # EVALUATE THE MODELS, OVER EPOCHS AND SAVE TO THE DATA DIRECTORY
    custom.evaluate_models(hp,h,fold_num,nn,epochs_completed,X_train,Y_train,X_val,Y_val,X_test,Y_test)


###############################################################################
# COLLECT ALL PERFORMANCE AND STORE IN ALL 'all_performance/'                                      
###############################################################################
#if testing == 0:
#    custom.compile_cross_fold_results(hp['number_of_folds'][h] ,epochs_completed,hp['experiment'][h] )


###############################################################################
#  REFERENCES 
###############################################################################
# Keras in the browser                               : https://github.com/transcranial/keras-js 
# Hyper-paramter otpimization                        : https://github.com/maxpumperla/hyperas
# Explaination of GANs                               : https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
# Information on Keras                               : https://github.com/fchollet/keras-resources
# A useful tutorial for beginners                    : https://dashee87.github.io/data%20science/deep%20learning/python/another-keras-tutorial-for-neural-network-beginners/
# Information on video classificaiton                : https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5
# Code on DNN video classification                   : https://github.com/harvitronix/five-video-classification-methods
# Paper on AUC optimization via SGM                  : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.2.3727&rep=rep1&type=pdf
# Tips to improve DNN performance                    : https://machinelearningmastery.com/improve-deep-learning-performance/
# Hyper-parameter optimzation information            : https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# Information on various Gradient Descent Algorithms : http://ruder.io/optimizing-gradient-descent/
# HELP CHOOSING INITIALIZATION OF WEIGHTS            : http://deepdish.io/2015/02/24/network-initialization/


# ----------------------------------------------------------------------------
#def inverse_binary_crossentropy(y_true, y_pred):
#    if theano.config.floatX == 'float64':
#        epsilon = 1.0e-9
#    else:
#        epsilon = 1.0e-7
#    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
#    bce = T.nnet.binary_crossentropy(y_pred, y_true).mean(axis=-1)
#    return -bce
# ----------------------------------------------------------------------------
# Siamese network objective 
# https://github.com/Lasagne/Lasagne/issues/168
#def hp['loss_function'][h] (y_true, y_pred):
#    a = y_pred[0::2]
#    b = y_pred[1::2]
#    diff = ((a - b) ** 2).sum(axis=1, keepdims=True)
#    y_true = y_true[0::2]
#    return ((diff - y_true)**2).mean()
