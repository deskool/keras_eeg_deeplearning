#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:30:54 2018

@author: Mohammad Ghassemi, PhD Candidate, MIT
"""
###############################################################################
#  REFERENCES 
###############################################################################
# A useful tutorial for beginners                   : https://dashee87.github.io/data%20science/deep%20learning/python/another-keras-tutorial-for-neural-network-beginners/
# Information on video classificaiton               : https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5
# Code on DNN video classification                  : https://github.com/harvitronix/five-video-classification-methods
# Paper on AUC optimization via SGM                 : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.2.3727&rep=rep1&type=pdf
# Tips to improve DNN performance                   : https://machinelearningmastery.com/improve-deep-learning-performance/
# Hyper-parameter optimzation information           : https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# Information on various Gradient Descent Algorithms: http://ruder.io/optimizing-gradient-descent/
# HELP CHOOSING INITIALIZATION OF WEIGHTS           : http://deepdish.io/2015/02/24/network-initialization/

###############################################################################
#IMPORT THE REQUISITE LIBRARIES
###############################################################################
#import pandas as pd
#import seaborn as sns
#import sys
#import glob
#import sklearn
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
#from keras.layers import Dense

import numpy as np
import random
import matplotlib.pyplot as plt
import pylab
import os
import sys
import json
import time
from sklearn.utils import class_weight
import keras
import keras.backend as K
from keras.models import Sequential
from keras import metrics 
import tensorflow as tf  

#Let the user know if they are not using a GPU
if K.tensorflow_backend._get_available_gpus() == []:
    print 'YOU ARE NOT USING A GPU ON THIS DEVICE!' 

#Check where the python executable is
sys.executable

#Check the path where it is grabbing its libraries
sys.path

#------------------------------------------------------------------------------
# Custom Functions
#------------------------------------------------------------------------------
import custom_evaluate                       
import custom_metrics                                  
import custom_callbacks
import custom_optimizer

reload(custom_optimizer)
reload(custom_callbacks)
reload(custom_metrics)                                
reload(custom_evaluate)

###############################################################################
# MODEL PARAMETERS
###############################################################################
#------------------------------------------------------------------------------
# Data Related Parameters 
#------------------------------------------------------------------------------
take_only         = .25                                # float, [0 1], % of the total data you want to keep
input_shape       = (72,8,8,4)                        # tuple, the shape of the input data
time_dir          = str(time.time())                  # string, Name of result directory set to "str(time.time())", for multiple runs 
 
#------------------------------------------------------------------------------
# Training Related Parameters 
#------------------------------------------------------------------------------
seed               =  1                              # Random Seed
train_perc         = .5                              # flaot, [0 1], Training percentage
val_perc           = .25                             # float, [0 1], Validation percentage
test_perc          = .25                             # float, [0 1], Test percentage
checkpoint_period  = 1                               # int,   >= 1 , Epochs before we have a checkpoint
number_of_folds    = 5                               # int,   >= 1 , Number of Cross validation folds
num_epochs         = 100                             # int,   >= 1 , Number of training data epochs
batch_perc         = .001                            # float, [0 1], Percentage of training to use data per training batch (1 = 100%)

#------------------------------------------------------------------------------
# Data Weighting parameters 
#------------------------------------------------------------------------------
balance_class_cost = False                           # bool , T|F  , weights data balance misclassificaiton costs.
class_weights      = [1, 1]                          # The weights of classes 0 and 1

#------------------------------------------------------------------------------
# General Neural Network Parameters
#------------------------------------------------------------------------------
# Initializers  : https://keras.io/initializers/
# Loss Functions: https://keras.io/losses/
weights_initialization = 'he_normal'                  # he_uniform|lecun_normal|he_normal|glorot_uniform|glorot_normal|lecun_uniform|Zeros|Ones|RandomNormal|RandomUniform|TruncatedNormal|VarianceScaling|Orthogonal
bias_initialization    = 'Zeros'                      # Options same as above. See all bias initializations 
loss_function          = 'binary_crossentropy'        # *custom*(see below)|binary_crossentropy|kullback_leibler_divergence|cosine_proximity|poisson|mean_squared_error|mean_absolute_error|mean_absolute_percentage_error|mean_squared_logarithmic_error|squared_hinge|hinge|categorical_hinge|logcosh|categorical_crossentropy|sparse_categorical_crossentropy|binary_crossentropy|
activation             = 'relu'                       # Activation function of the network
                                                    
#------------------------------------------------------------------------------
# Optimizers parameters [custom_optimizer.py]: 
#------------------------------------------------------------------------------
# Optimizers: https://keras.io/optimizers/ 
# Use [] for defaults 
optimizer              = 'adam'                    # sgd|rmsprop|adagrad|adadelta|adam|adamaz|nadam
lr                     = []                         # float, >= 0           
decay                  = []                         # float, >= 0
momentum               = []                         # float, >= 0
rho                    = []                         # float, >= 0
epsilon                = []                         # float, >= 0
beta_1                 = []                         # float, 0 < beta < 1
beta_2                 = []                         # float, 0 < beta < 1
nesterov               = []                         # bool , T|F
amsgrad                = []                         # bool , T|F

#------------------------------------------------------------------------------
# Callback Paramters [custom_metrics.py]:
#------------------------------------------------------------------------------
# Callback parameters for 'Metrics': https://keras.io/metrics/                          
selected_metrics       = [metrics.binary_accuracy] 
  
# Callback parameters for 'Early Stopping': https://keras.io/callbacks/
early_stop_min_delta  = 0                           # minimum amount things must change before we start applying the patience.
early_stop_patience   = 100                         # number of epochs with no improvement after which training will be stopped.

# Callback parameters for 'Reduce Learning Rate': https://keras.io/callbacks/
reduce_lr_patience    = 5                           # How long must performance platteaus before lr is reduced  
reduce_lr_factor      = 0.2                         # Amount tot reduce the loarning rate by: .2 is a factor of 5 
reduce_lr_epsilon     = 0.0001                      # threshold for measuring new optimum
reduce_lr_cooldown    = 3                           # Number ofepochs to keep at this lr before resuming normal activity 
reduce_lr_min_lr      = 0.001                       # minimum learning rate we can reduce to


#------------------------------------------------------------------------------
# Generate Directories for the cross-fold validation   
#------------------------------------------------------------------------------         
custom_evaluate.gen_folders(time_dir,number_of_folds)      
                      
# ----------------------------------------------------------------------------
# THE CUSTOM LOSS FUNCTION - USED WHEN: loss_function = 'custom'               
# -----------------------------------------------------------------------------                                
# https://www.kaggle.com/tomcwalker/keras-nn-with-custom-loss-function-for-gini-auc/notebook
#This function is meant to optimize AUC, but it doesn't work very well.
if loss_function == 'custom':
    def loss_function(y_true, y_pred):
        y_true   = tf.cast(y_true, tf.int32)
        parts    = tf.dynamic_partition(y_pred, y_true, 2)
        y_pos    = parts[1]
        y_neg    = parts[0]
        y_pos    = tf.expand_dims(y_pos, 0)
        y_neg    = tf.expand_dims(y_neg, -1)
        out      = K.sigmoid(y_neg - y_pos)
        return K.mean(out)

#------------------------------------------------------------------------------
# Specify the network - Simple Multilayer
#------------------------------------------------------------------------------
#nn = Sequential()                                     
#nn.add(Dense(10,                                          # 8 input -> 10 hidden layer
#             input_shape        = (8,), 
#             activation         = activation,
#             kernel_initializer = weights_initialization,
#             bias_initializer   = bias_initialization
#             )
#       ) 
#nn.add(Dense(6,                                            # 10 hidden layer -> 6 hidden layer
#             activation         = activation,
#             kernel_initializer = weights_initialization,
#             bias_initializer   = bias_initialization
#             )
#       ) 
#nn.add(Dense(1, activation='sigmoid'))                      # Outcome, 1/0

#------------------------------------------------------------------------------
# Specify the network - 3D CNN
# Inspired by: https://github.com/kcct-fujimotolab/3DCNN/blob/master/3dcnn.py
#------------------------------------------------------------------------------
nn = Sequential()
# 3D Convolution with 8, 3x3x3 filters
nn.add(keras.layers.Conv3D(8,           
                           kernel_size = (3, 3, 3) ,
                           strides=(1, 1, 1),
                           kernel_initializer = weights_initialization,
                           input_shape=input_shape, 
                           data_format='channels_last',
                           padding='same'))
# Relu Activation
nn.add(keras.layers.Activation('relu'))
# 3D Convolution with 8, 3x3x3 filters
nn.add(keras.layers.Conv3D(8, 
                           kernel_size=(3, 3, 3),
                           strides=(1, 1, 1),
                           kernel_initializer = weights_initialization,
                           padding='same'))
# Softmax activation
nn.add(keras.layers.Activation('softmax'))
# 3D Max Pooling
nn.add(keras.layers.MaxPooling3D(pool_size=(3, 3, 3), 
                                 padding='same'))
nn.add(keras.layers.Dropout(0.25))
# Dropout
nn.add(keras.layers.Conv3D(16, 
                           kernel_size=(3, 3, 3),
                           strides=(1, 1, 1),
                           kernel_initializer = weights_initialization,
                           padding='same'))
# Activation
nn.add(keras.layers.Activation('relu'))
# 3D Convolution
nn.add(keras.layers.Conv3D(16, 
                           kernel_size=(3, 3, 3),
                           strides=(1, 1, 1),
                           kernel_initializer = weights_initialization, 
                           padding='same'))
nn.add(keras.layers.Activation('softmax'))
# 3D Max Pooling
nn.add(keras.layers.MaxPooling3D(pool_size=(3, 3, 3), 
                                 padding='same'))
# Dropout
nn.add(keras.layers.Dropout(0.25))
#Flatten
nn.add(keras.layers.Flatten())
# Dense
nn.add(keras.layers.Dense(128, 
                          activation='sigmoid'))
# Dropout
nn.add(keras.layers.Dropout(0.5))
# Dense
nn.add(keras.layers.Dense(1, 
                          activation='sigmoid'))

# Save the Neural Network Topology -------------------------------------------
json_nn = nn.to_json()
with open('Checkpoints/' + time_dir + '/topology/model.json', 'w') as outfile:
    json.dump(json_nn, outfile)

#------------------------------------------------------------------------------
# Save the network hyperparameters 
#------------------------------------------------------------------------------
hyper_parameters = {'take_only'  : take_only,
                    'seed'       : seed,
                   'train_perc': train_perc,
                   'val_perc'  : val_perc,
                   'test_perc' :test_perc,
                   'checkpoint_period' : checkpoint_period,
                   'number_of_folds'   : number_of_folds,
                   'num_epochs' : num_epochs,
                   'batch_perc' : batch_perc,
                   'balance_class_cost' : balance_class_cost,
                   'class_weights' : class_weights,
                   'weights_initialization' : weights_initialization,
                   'bias_initialization' : bias_initialization,
                   'loss_function' : loss_function,
                   'activation' : activation,
                   'optimizer' : optimizer,
                   'lr'       : lr,
                   'decay'    : decay,
                   'momentum' : momentum,
                   'rho'      : rho,
                   'epsilon'  : epsilon,
                   'beta_1'   : beta_1,
                   'beta_2'   : beta_2,
                   'nesterov' : nesterov,
                   'amsgrad'  : amsgrad,
                   'selected_metrics' : selected_metrics,
                   'early_stop_min_delta' : early_stop_min_delta,
                   'early_stop_patience' : early_stop_patience,
                   'reduce_lr_patience' : reduce_lr_patience,
                   'reduce_lr_factor' : reduce_lr_factor,
                   'reduce_lr_epsilon' : reduce_lr_epsilon,
                   'reduce_lr_cooldown' : reduce_lr_cooldown,
                   'reduce_lr_min_lr' : reduce_lr_min_lr}

#save the hyper-paramters of the network
np.save('Checkpoints/'  + time_dir + '/topology/hyper_parameters.npy',hyper_parameters)  

###############################################################################
# LOAD THE TRAINING AND TESTING DATA 
###############################################################################
np.random.seed(seed)      

# load in the EEG data...
covars    = np.load('python_covariates/covars.npy')
header    = np.load('python_covariates/covars_headers.npy')
files     = np.load('python_covariates/python_file_locs.npy')
inst      = np.load('python_covariates/inst.npy')
outcomes  = covars[:,-1:]

eg_data = np.load(files[0])
n_patients = np.size(files,0)
n_samples = np.size(eg_data,0)
n_frames = np.size(eg_data,1)
row = np.size(eg_data,2)
col = np.size(eg_data,3)
n_hertz = np.size(eg_data,4)
del eg_data

# Get the list of unique institutions
u_inst = np.unique(inst)

#let's try leave-one-inst-out:
a = []

for fold_num in range(0,np.size(u_inst)):

    #find the index of all good and bad outcomes
    bad_out_ind  = pylab.find(outcomes == 1)
    good_out_ind = pylab.find(outcomes == 0)
    
    # Get all subjects from a given 'inst' and make them the 'test' set.
    test_ind       = pylab.find(inst == u_inst[fold_num])
    test_perc      = 1.0*np.size(test_ind)/np.size(outcomes) 
    X_test,Y_test  = custom_evaluate.gen_data(test_ind,files,covars[test_ind,-1],take_only)
    
    #remove the test subjects from the  from the trianing/test set.
    bad_out_ind  =  np.array(list(set(bad_out_ind) - set(test_ind) ))
    good_out_ind = np.array(list(set(good_out_ind) - set(test_ind) ))

    #Take a random 50% of the remaining subjects and split them into training and validation sets.
    train_bad_out_ind    = random.sample(bad_out_ind, int(np.size(bad_out_ind)*(train_perc/(val_perc+train_perc))))
    train_good_out_ind   = random.sample(good_out_ind, int(np.size(good_out_ind)*(train_perc/(val_perc+train_perc))))
    train_ind            = np.append(train_bad_out_ind,train_good_out_ind)
    X_train,Y_train      = custom_evaluate.gen_data(train_ind,files,covars[train_ind,-1],take_only)
        
    #Leave what's left over for validation.
    val_bad_out_ind    = np.array(list(set(bad_out_ind) - set(train_bad_out_ind)))
    val_good_out_ind   = np.array(list(set(good_out_ind) - set(train_good_out_ind)))
    val_ind            = np.append(val_bad_out_ind,val_good_out_ind)
    X_val,Y_val        = custom_evaluate.gen_data(val_ind,files,covars[val_ind,-1],take_only)


    # load pima indians dataset
    #dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data", header=None).values
    #data    = dataset[:,0:8]
    #outcome = dataset[:,8] 
                          
    # Scale the data using Sklearn tools
    # scaler = sklearn.preprocessing.StandardScaler()                     
    # data = scaler.fit_transform(data)

    #Split the data into training testing and validation sets
    #[X_train, X_val, X_test, Y_train, Y_val, Y_test] = custom_evaluate.split_bin_data(data,outcome,train_perc,val_perc,test_perc)

    #Determine the batch size for NN training
    batch_size = int(round(batch_perc*np.size(X_train,0)))

    # Generate weights for the training data
    if balance_class_cost == True:
        class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    elif balance_class_cost == False:
        if type(class_weights) == list:
         class_weights = {0 : class_weights[0],
                          1 : class_weights[1]} 
    
    #clean up
    #del [dataset,data,outcome,test_perc,train_perc,val_perc]
     
    ###############################################################################
    # USE PARAMETERS TO INITIALIZE OPTIMIZATION ALGORITHM              
    ###############################################################################       
    # Optimizer Selection using parameters
    grad_desc_algorithm = custom_optimizer.choose_optimizer(optimizer,lr,decay,momentum,rho,epsilon,beta_1,beta_2,nesterov,amsgrad)  
        
    # Specfify model callbacks using parameters
    [checkpoint,terminate_on_nan,early_stopping,learning_rate] = custom_callbacks.callbacks(time_dir + '/fold' + str(fold_num),checkpoint_period,early_stop_min_delta,early_stop_patience,reduce_lr_factor,reduce_lr_patience,reduce_lr_epsilon,reduce_lr_cooldown,reduce_lr_min_lr)  
    
    # compile the network
    nn.compile(loss = loss_function, optimizer = grad_desc_algorithm, metrics = selected_metrics)
    
    ###############################################################################
    # FIT THE MODELS
    ###############################################################################
    # Fit the model: https://keras.io/models/model/#fit
    fit_nn = nn.fit(X_train,                          # Training Data X
                    Y_train,                          # Training Data Y
                    validation_data = (X_val,Y_val),  # Validation data tuple
                    shuffle         = 1,              # shuffle the training data epoch before you use it
                    initial_epoch   = 0,              # Starting Epoch (should awlways be 0)
                    epochs          = num_epochs,     # Number of runs through the data 
                    batch_size      = batch_size,     # Number of samples per gradient update. 
                    verbose         = 1,              # display options to console
                    class_weight    = class_weights,  # Force balanced class penalties 
                    callbacks=[checkpoint,            # We want to save checkpoints
                               terminate_on_nan,      # terminate run if nans are found                      
                               learning_rate,         # How to change learning rate.
                               early_stopping,        # The criteria for early stopping 
                               ])                    
    
    #Count the number of epochs that were completed before the convergence criteria
    epochs_completed = len(fit_nn.history['loss'])
   
    
    ###############################################################################
    # EVALUATE THE MODELS, OVER EPOCHS...
    ###############################################################################
    temp_test_model = nn
    train_over_time = []
    val_over_time   = []
    test_over_time  = []
    
    #For each n epochs...
    for i in xrange(checkpoint_period,epochs_completed,checkpoint_period):
        nn.load_weights("Checkpoints/" + time_dir + '/fold' + str(fold_num) + "/weights/nn_weights-%02d.hdf5" % i)
    
        #EVALUATE TRAINING SET 
        loss = nn.evaluate(X_train, Y_train, verbose=0)
        y_pred = nn.predict(X_train,verbose=0)
        [test_names, results] = custom_evaluate.evaluate_model(Y_train, y_pred)
        results = np.append(loss[0],np.array(results))
        train_over_time.append(results)
        
        #EVALUATE VALIDATION SET
        loss = nn.evaluate(X_val, Y_val, verbose=0)
        y_pred = nn.predict(X_val,verbose=0)
        [test_names, results] = custom_evaluate.evaluate_model(Y_val, y_pred)
        results = np.append(loss[0],np.array(results))
        val_over_time.append(results)
        
        #EVALUATE TESTING SET
        loss = nn.evaluate(X_test, Y_test, verbose=0)
        y_pred = nn.predict(X_test,verbose=0)
        [test_names, results] = custom_evaluate.evaluate_model(Y_test, y_pred)
        results = np.append(loss[0],np.array(results))
        test_over_time.append(results)
    
    train_over_time = np.vstack(train_over_time)
    val_over_time = np.vstack(val_over_time)  
    test_over_time = np.vstack(test_over_time)    
    test_names.insert(0,'Loss')
    
    # SAVE THE RESULTS ############################################################
    np.save('Checkpoints/' + time_dir + '/fold' + str(fold_num) +'/performance/train_performance',train_over_time)
    np.save('Checkpoints/' + time_dir + '/fold' + str(fold_num) +'/performance/val_performance',val_over_time) 
    np.save('Checkpoints/' + time_dir + '/fold' + str(fold_num) +'/performance/test_performance',test_over_time)
    np.save('Checkpoints/' + time_dir + '/fold' + str(fold_num) +'/performance/metric_names',test_names)      
    
    ###############################################################################
    # PLOT PERFORMANCE OVER TIME ...                                              
    ############################################################################### 
    xdim = 5
    ydim = 7
    
    # generate the plots
    fig,axarr = plt.subplots(xdim, ydim,figsize=(16, 18), dpi=100)
    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9 , top=0.9, wspace=0.4, hspace=0.4)
    
    ind = 0
    for i in range(0,xdim):
        for j in range(0,ydim):
            if ind < np.size(train_over_time,1):
                #Plot the training
                axarr[i,j].plot(xrange(checkpoint_period,epochs_completed,checkpoint_period), 
                                train_over_time[:,ind],
                                linestyle='-', 
                                color='blue',
                                label='Training', 
                                lw=2)
                #Plot the validation
                axarr[i,j].plot(xrange(checkpoint_period,epochs_completed,checkpoint_period), 
                                val_over_time[:,ind],
                                linestyle='-', 
                                color='red',
                                label='Validation', 
                                lw=2)
                #Plot the testing 
                axarr[i,j].plot(xrange(checkpoint_period,epochs_completed,checkpoint_period), 
                                test_over_time[:,ind],
                                linestyle='-', 
                                color='green',
                                label='Test', 
                                lw=2)
                
                #Title the plots according to the names
                axarr[i,j].set_title(test_names[ind], fontsize=8)
                
                #Strip off the boxes
                axarr[i,j].spines['top'].set_visible(False)
                axarr[i,j].spines['right'].set_visible(False)
                axarr[i,j].spines['bottom'].set_visible(False)
                axarr[i,j].spines['left'].set_visible(False)
                
                #If this is the last row of plots
                if ind >= (xdim*ydim)-ydim:
                    axarr[i,j].set_xlabel('Training Epoch')
                else:
                    axarr[i,j].set_xticks([])
                    axarr[i,j].set_xticklabels([])
            
            #If we've run out of data clear the plots
            elif ind >= np.size(train_over_time,1):
                leg = axarr[i,j-1].legend(bbox_to_anchor=(1.3, .75), loc=2, borderaxespad=0.,fontsize=13)
                axarr[i,j].set_xticklabels([])
                axarr[i,j].set_xticks([])
                axarr[i,j].set_visible(False)
            # Stick the legend in the final plot
            
            ind = ind +1
    
    plt.savefig('Checkpoints/' + time_dir + '/fold' + str(fold_num) +'/performance/perf.png')
 
###############################################################################
# COLLECT ALL PERFORMANCE AND STORE IN ALL 'all_performance/'                                      
############################################################################### 
all_train = np.zeros([number_of_folds,epochs_completed-1,34])
all_val   = np.zeros([number_of_folds,epochs_completed-1,34])
all_test   = np.zeros([number_of_folds,epochs_completed-1,34])

for i in range(0,number_of_folds):
    all_train[i,:,:] = np.load('Checkpoints/' + time_dir + '/fold' + str(i) +'/performance/train_performance.npy')
    all_val[i,:,:]   = np.load('Checkpoints/' + time_dir + '/fold' + str(i) +'/performance/val_performance.npy')
    all_test[i,:,:]  = np.load('Checkpoints/' + time_dir + '/fold' + str(i) +'/performance/test_performance.npy')


np.save('Checkpoints/' + time_dir + '/all_performance/all_train',all_train)
np.save('Checkpoints/' + time_dir + '/all_performance/all_val',all_val)
np.save('Checkpoints/' + time_dir + '/all_performance/all_test',all_test)

###############################################################################
# Siamese network objective 
# https://github.com/Lasagne/Lasagne/issues/168
#def loss_function(y_true, y_pred):
#    a = y_pred[0::2]
#    b = y_pred[1::2]
#    diff = ((a - b) ** 2).sum(axis=1, keepdims=True)
#    y_true = y_true[0::2]
#    return ((diff - y_true)**2).mean()
