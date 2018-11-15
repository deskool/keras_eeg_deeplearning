#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 13:26:50 2018

@author: mohammad
"""


#MODEL CHECKPOINTS
import keras
from keras.callbacks import ModelCheckpoint
from time import time
import sklearn
import numpy as np
import pylab
from sklearn.model_selection import train_test_split
import matplotlib as plt
import os
import json
import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPooling2D,Dropout,LSTM,Conv3D,MaxPooling3D,Conv2D,TimeDistributed,BatchNormalization
from sklearn.utils import class_weight

def convert4D_to_2D(X_train,Y_train, X_val,Y_val,X_test,Y_test):
    # Train -----------------------------------------------------------------------
    ds = X_train.shape
    temp = X_train.reshape(ds[0]*ds[1],ds[2],ds[3],ds[4])
    temp_out = np.repeat(Y_train,ds[1],0)
    
    keep_list = []
    for i in range(temp.shape[0]):
        if sum(sum(sum(temp[i,:,:,:]))) != 0:
            keep_list.append(i)
    
    X_train = temp[keep_list,:,:,:]
    Y_train = temp_out[keep_list]      
    
    # Vali ------------------------------------------------------------------------
    ds = X_val.shape
    temp = X_val.reshape(ds[0]*ds[1],ds[2],ds[3],ds[4])
    temp_out = np.repeat(Y_val,ds[1],0)
    
    keep_list = []
    for i in range(temp.shape[0]):
        if sum(sum(sum(temp[i,:,:,:]))) != 0:
            keep_list.append(i)
    
    X_val = temp[keep_list,:,:,:]
    Y_val = temp_out[keep_list]    
    
    # Test ------------------------------------------------------------------------
    ds = X_test.shape
    temp = X_test.reshape(ds[0]*ds[1],ds[2],ds[3],ds[4])
    temp_out = np.repeat(Y_test,ds[1],0)
    
    keep_list = []
    for i in range(temp.shape[0]):
        if sum(sum(sum(temp[i,:,:,:]))) != 0:
            keep_list.append(i)
    
    X_test = temp[keep_list,:,:,:]
    Y_test = temp_out[keep_list]
    
    return X_train,Y_train, X_val,Y_val,X_test,Y_test


def evaluate_model(y_true, y_pred):
    #Area under PR curve
        
    [p, r, th]              = sklearn.metrics.precision_recall_curve(y_true, y_pred) # GET THE AUC 
    test_auc_prc            = sklearn.metrics.auc(r, p)
    
    #plt.plot(r,p)

    #best recall for various levels of percision
    r_at_p100                 = np.max(r[pylab.find(p >= 1)])                    # find r | p >= 0.00 
    r_at_p99                  = np.max(r[pylab.find(p >= 0.99)])                 # find tpr | fpr <= 0.01 
    r_at_p95                  = np.max(r[pylab.find(p >= 0.95)])                 # find tpr | fpr <= 0.05
    
    # Area under the ROC curve
    test_auc_roc               = sklearn.metrics.roc_auc_score(y_true,y_pred)     # get the AUC ROC
    
    # TPR values at various FPR values
    [fpr, tpr, thresholds]     = sklearn.metrics.roc_curve(y_true, y_pred)        # Generate the curve
    tpr_at_fpr0                = np.max(tpr[pylab.find(fpr <= 0.00)])             # find tpr | fpr <= 0.00 
    tpr_at_fpr1                = np.max(tpr[pylab.find(fpr <= 0.01)])             # find tpr | fpr <= 0.01 
    tpr_at_fpr5                = np.max(tpr[pylab.find(fpr <= 0.05)])             # find tpr | fpr <= 0.05
        
    #plt.plot(fpr,tpr)
    
    # Classifier Thresholds at various FRP values
    threshold_at_fpr0           = np.min(thresholds[pylab.find(fpr <= 0.00)])      # classifier threshold for fpr <= 0.00
    threshold_at_fpr1           = np.min(thresholds[pylab.find(fpr <= 0.01)])      # classifier threshold for fpr <= 0.01
    threshold_at_fpr5           = np.min(thresholds[pylab.find(fpr <= 0.05)])      # classifier threshold for fpr <= 0.05   
    
    # Accuracy
    test_accuracy               = sklearn.metrics.accuracy_score(y_true, 1*(y_pred > 0.5))
    test_accuracy_at_fpr0       = sklearn.metrics.accuracy_score(y_true, 1*(y_pred > threshold_at_fpr0))
    test_accuracy_at_fpr1       = sklearn.metrics.accuracy_score(y_true, 1*(y_pred > threshold_at_fpr1))
    test_accuracy_at_fpr5       = sklearn.metrics.accuracy_score(y_true, 1*(y_pred > threshold_at_fpr5))
    
    # FScore
    test_f1score                = sklearn.metrics.f1_score(y_true, 1*(y_pred > 0.5))
    test_f1_at_fpr0             = sklearn.metrics.f1_score(y_true, 1*(y_pred > threshold_at_fpr0))
    test_f1_at_fpr1             = sklearn.metrics.f1_score(y_true, 1*(y_pred > threshold_at_fpr1))
    test_f1_at_fpr5             = sklearn.metrics.f1_score(y_true, 1*(y_pred > threshold_at_fpr5))
            
    # Jaccard Simmilarity
    test_jaccard_simm           = sklearn.metrics.jaccard_similarity_score(y_true, 1*(y_pred > 0.5))
    test_jaccard_simm_at_fpr0   = sklearn.metrics.jaccard_similarity_score(y_true, 1*(y_pred > threshold_at_fpr0))
    test_jaccard_simm_at_fpr1   = sklearn.metrics.jaccard_similarity_score(y_true, 1*(y_pred > threshold_at_fpr1))
    test_jaccard_simm_at_fpr5   = sklearn.metrics.jaccard_similarity_score(y_true, 1*(y_pred > threshold_at_fpr5))
    
    # Matthres Correlation Coefficient
    test_matthres_corrcoef              = sklearn.metrics.matthews_corrcoef(y_true, 1*(y_pred > 0.5))
    test_matthres_corrcoef_at_fpr0      = sklearn.metrics.matthews_corrcoef(y_true, 1*(y_pred > threshold_at_fpr0))
    test_matthres_corrcoef_at_fpr1      = sklearn.metrics.matthews_corrcoef(y_true, 1*(y_pred > threshold_at_fpr1))
    test_matthres_corrcoef_at_fpr5      = sklearn.metrics.matthews_corrcoef(y_true, 1*(y_pred > threshold_at_fpr5))
    
    # Hamming Loss
    test_hamming_loss           = sklearn.metrics.hamming_loss(y_true, 1*(y_pred > 0.5))
    test_hamming_loss_at_fpr0   = sklearn.metrics.hamming_loss(y_true, 1*(y_pred > threshold_at_fpr0))
    test_hamming_loss_at_fpr1   = sklearn.metrics.hamming_loss(y_true, 1*(y_pred > threshold_at_fpr1))
    test_hamming_loss_at_fpr5   = sklearn.metrics.hamming_loss(y_true, 1*(y_pred > threshold_at_fpr5))
      
    #Logg Logg
    test_log_loss           = sklearn.metrics.log_loss(y_true, y_pred)           # Log Loss
    
    #The Brier score: a proper score function that measures the accuracy of probabilistic predictions. 
    test_breir_score        = sklearn.metrics.brier_score_loss(y_true, y_pred)  #Brier Score
    

    return [['AUC(PRC)', 
            'R|P=100%', 
            'R|P=99%', 
            'R|P=95%', 
            'AUC(ROC)', 
            'TPR|FPR=0%', 
            'TPR|FPR=1%', 
            'TPR|FPR=5%',
            'Threshold|FPR=0%', 
            'Threshold|FPR=1%', 
            'Threshold|FPR=5%',
            'Accuracy', 
            'Accuracy|FPR=0%',
            'Accuracy|FPR=1%',
            'Accuracy|FPR=5%',
            'F1 Score', 
            'F1 Score|FPR=0%',
            'F1 Score|FPR=1%',
            'F1 Score|FPR=5%',
            'Jaccard Simm',
            'Jaccard Simm|FPR=0%',
            'Jaccard Simm|FPR=1%',
            'Jaccard Simm|FPR=5%',
            'Matthres Corr',
            'Matthres Corr|FPR=0%',
            'Matthres Corr|FPR=1%',
            'Matthres Corr|FPR=5%',
            'Hamming Loss',
            'Hamming Loss|FPR=0%',
            'Hamming Loss|FPR=1%',
            'Hamming Loss|FPR=5%',
            'Log Loss',
            'Breir Score'
            ],
            [test_auc_prc, 
             r_at_p100, 
             r_at_p99, 
             r_at_p95, 
             test_auc_roc, 
             tpr_at_fpr0, 
             tpr_at_fpr1, 
             tpr_at_fpr5,
             threshold_at_fpr0, 
             threshold_at_fpr1, 
             threshold_at_fpr5,
             test_accuracy, 
             test_accuracy_at_fpr0,
             test_accuracy_at_fpr1,
             test_accuracy_at_fpr5,
             test_f1score, 
             test_f1_at_fpr0, 
             test_f1_at_fpr1, 
             test_f1_at_fpr5,
             test_jaccard_simm,
             test_jaccard_simm_at_fpr0,
             test_jaccard_simm_at_fpr1,
             test_jaccard_simm_at_fpr5,
             test_matthres_corrcoef,
             test_matthres_corrcoef_at_fpr0,
             test_matthres_corrcoef_at_fpr1,
             test_matthres_corrcoef_at_fpr5,
             test_hamming_loss,
             test_hamming_loss_at_fpr0,
             test_hamming_loss_at_fpr1,
             test_hamming_loss_at_fpr5,
             test_log_loss,
             test_breir_score]]


def evaluate_models(hp,h,fold_num,nn,epochs_completed,X_train,Y_train,X_val,Y_val,X_test,Y_test):

    ###############################################################################
    # EVALUATE THE MODELS, OVER EPOCHS...
    ###############################################################################
    train_over_time = []
    val_over_time   = []
    test_over_time  = []
    
    #For each n epochs...
    for i in range(hp['checkpoint_period'][h] ,epochs_completed,hp['checkpoint_period'][h] ):
        nn.load_weights("Checkpoints/" + hp['experiment'][h]  + '/fold' + str(fold_num) + "/weights/nn_weights-%02d.hdf5" % i)
        
        #EVALUATE TRAINING SET 
        loss = nn.evaluate(X_train, Y_train, verbose=0)
        y_pred = nn.predict(X_train,verbose=0)
        [test_names, results] = evaluate_model(Y_train, y_pred)
        results = np.append(loss[0],np.array(results))
        train_over_time.append(results)
        
        #EVALUATE VALIDATION SET
        loss = nn.evaluate(X_val, Y_val, verbose=0)
        y_pred = nn.predict(X_val,verbose=0)
        [test_names, results] = evaluate_model(Y_val, y_pred)
        results = np.append(loss[0],np.array(results))
        val_over_time.append(results)
        
        #EVALUATE TESTING SET
        loss = nn.evaluate(X_test, Y_test, verbose=0)
        y_pred = nn.predict(X_test,verbose=0)
        [test_names, results] = evaluate_model(Y_test, y_pred)
        results = np.append(loss[0],np.array(results))
        test_over_time.append(results)
    
    train_over_time = np.vstack(train_over_time)
    val_over_time = np.vstack(val_over_time)  
    test_over_time = np.vstack(test_over_time)    
    test_names.insert(0,'Loss')
    
    # SAVE THE RESULTS ############################################################
    np.save('Checkpoints/' + hp['experiment'][h]  + '/fold' + str(fold_num) +'/performance/train_performance',train_over_time)
    np.save('Checkpoints/' + hp['experiment'][h]  + '/fold' + str(fold_num) +'/performance/val_performance',val_over_time) 
    np.save('Checkpoints/' + hp['experiment'][h]  + '/fold' + str(fold_num) +'/performance/test_performance',test_over_time)
    np.save('Checkpoints/' + hp['experiment'][h]  + '/fold' + str(fold_num) +'/performance/metric_names',test_names)      
    
    ###############################################################################
    # PLOT PERFORMANCE OVER TIME ...                                              
    ############################################################################### 
    #if plot_perf == True:
    #    plot_performance(hp['time_dir'][h] ,fold_num,train_over_time,val_over_time,test_over_time,test_names,epochs_completed,hp['checkpoint_period'][h] )


def  LRCN(hp,h):
    model = Sequential()

    for i in range(0,hp['lrcn_num_cnn_layers'][h]):

        #Convolution
        if i == 0:
            model.add(TimeDistributed(Conv2D(hp['lrcn_cnn_filters_per_layer'][h][i],
                                             hp['lrcn_cnn_kernal_size_per_layer'][h][i], 
                                             strides            = hp['lrcn_cnn_strides_per_layer'][h][i],
                                             kernel_initializer = hp['weights_initialization'][h],
                                             bias_initializer   = hp['bias_initialization'][h], 
                                             padding            = 'same', 
                                             data_format        = 'channels_last'), 
                                             input_shape        = hp['input_shape'][h])) 

        else: 
            model.add(TimeDistributed(Conv2D(hp['lrcn_cnn_filters_per_layer'][h][i],
                                             hp['lrcn_cnn_kernal_size_per_layer'][h][i], 
                                             strides            = hp['lrcn_cnn_strides_per_layer'][h][i],
                                             kernel_initializer = hp['weights_initialization'][h],
                                             bias_initializer   = hp['bias_initialization'][h], 
                                             padding            = 'same'))) 

        #Batch Noramlization
        if hp['lrcn_batchnormalize_per_layer'][h][i] != 0:
            model.add(TimeDistributed(BatchNormalization(axis = hp['lrcn_batchnormalize_axis'][h], 
                                momentum               = hp['lrcn_batchnormalize_momentum'][h], 
                                epsilon                = hp['lrcn_batchnormalize_epsilon'][h], 
                                center                 = hp['lrcn_batchnormalize_center'][h], 
                                scale                  = hp['lrcn_batchnormalize_scale'][h], 
                                beta_initializer       = hp['lrcn_batchnormalize_beta_init'][h], 
                                gamma_initializer      = hp['lrcn_batchnormalize_gamma_init'][h], 
                                moving_mean_initializer = hp['lrcn_batchnormalize_moving_mean'][h], 
                                moving_variance_initializer =hp['lrcn_batchnormalize_moving_var'][h], 
                                beta_regularizer       = None, 
                                gamma_regularizer      = None, 
                                beta_constraint        = None, 
                                gamma_constraint       = None)))




        #Activation
        if hp['lrcn_activation_type_per_layer'][h][i] != []:
            model.add(TimeDistributed(Activation(hp['lrcn_activation_type_per_layer'][h][i])))               

        # Maxpooling
        if hp['lrcn_cnn_poolsize_per_layer'][h][i] != []:
            model.add(TimeDistributed(MaxPooling2D(hp['lrcn_cnn_poolsize_per_layer'][h][i], 
                                           strides=hp['lrcn_cnn_poolstrides_per_layer'][h][i] )))

        #Dropouts
        if hp['lrcn_dropouts_per_layer'][h][i] != []:
            model.add(TimeDistributed(Dropout(hp['lrcn_dropouts_per_layer'][h][i])))


    #Flatten 
    model.add(TimeDistributed(keras.layers.Flatten()))

    #Final Dropout
    if hp['lrcn_pre_lstm_dropout'][h] != []:
        model.add(Dropout(hp['lrcn_pre_lstm_dropout'][h]))

    #LSTM 
    model.add(LSTM(hp['lrcn_lstm_output_dim'][h], 
                   activation            = hp['lrcn_lstm_activation'][h], 
                   recurrent_activation  = hp['lrcn_lstm_recurrent_activation'][h], 
                   use_bias              = hp['lrcn_lstm_use_bias'][h], 
                   kernel_initializer    = hp['lrcn_ltsm_kernal_initializer'][h], 
                   recurrent_initializer = hp['lrcn_lstm_recurrent_initializer'][h], 
                   bias_initializer      = hp['lrcn_lstm_bias_intitializer'][h], 
                   dropout               = hp['lrcn_lstm_dropout'][h], 
                   recurrent_dropout     = hp['lrcn_lstm_recurrent_dropout'][h],  
                   return_sequences      = False, 
                   return_state          = False, 
                   go_backwards          = False, 
                   stateful              = False, 
                   unroll                = False))
    
    #Final activation
    model.add(Dense(1, activation      = 'sigmoid'))

    return model

def CNN_3D(hp,h):
    
    nn = Sequential()

    for i in range(0,hp['3dcnn_num_layers'][h]):
        #--------------------------------------------------------------------------
        # 3D Convolution
        if i == 0:
            nn.add(Conv3D(hp['3dcnn_filters_per_layer'][h][i],                        # 8
                                       kernel_size        = hp['3dcnn_kernal_sizes_per_layer'][h][i],     # (3, 3, 3),
                                       strides            = hp['3dcnn_strides_per_layer'][h][i],          # (1, 1, 1),
                                       kernel_initializer = hp['weights_initialization'][h],
                                       bias_initializer   = hp['bias_initialization'][h],
                                       input_shape        = hp['input_shape'][h], 
                                       data_format        ='channels_last',
                                       padding            ='same'))
        else:
            nn.add(Conv3D(hp['3dcnn_filters_per_layer'][h][i],                        # 8
                                       kernel_size        = hp['3dcnn_kernal_sizes_per_layer'][h][i],     # (3, 3, 3),
                                       strides            = hp['3dcnn_strides_per_layer'][h][i],          # (1, 1, 1),
                                       kernel_initializer = hp['weights_initialization'][h],
                                       bias_initializer   = hp['bias_initialization'][h],
                                       padding            = 'same'))

        
        #Batch Noramlization
        if hp['3dcnn_batchnormalize_per_layer'][h][i] != []:
            nn.add(BatchNormalization( axis            = hp['3dcnn_batchnormalize_axis'][h], 
                                momentum               = hp['3dcnn_batchnormalize_momentum'][h], 
                                epsilon                = hp['3dcnn_batchnormalize_epsilon'][h], 
                                center                 = hp['3dcnn_batchnormalize_center'][h], 
                                scale                  = hp['3dcnn_batchnormalize_scale'][h], 
                                beta_initializer       = hp['3dcnn_batchnormalize_beta_init'][h], 
                                gamma_initializer      = hp['3dcnn_batchnormalize_gamma_init'][h], 
                                moving_mean_initializer = hp['3dcnn_batchnormalize_moving_mean'][h], 
                                moving_variance_initializer =hp['3dcnn_batchnormalize_moving_var'][h], 
                                beta_regularizer       = None, 
                                gamma_regularizer      = None, 
                                beta_constraint        = None, 
                                gamma_constraint       = None))


        #Activation
        if hp['3dcnn_activations_per_layer'][h][i] != []:
            nn.add(Activation(hp['3dcnn_activations_per_layer'][h][i]))               # relu
        
        #3D Max pooling
        if hp['3dcnn_poolsize_per_layer'][h][i] != []:
            nn.add(MaxPooling3D(pool_size   = hp['3dcnn_poolsize_per_layer'][h][i],    # (3, 3, 3), 
                                             padding     = 'same'))
        
        # Dropout
        if hp['3dcnn_dropouts_per_layer'][h][i] != []:
            nn.add(Dropout(hp['3dcnn_dropouts_per_layer'][h][i]))                            # 0.025


    #Flatten
    nn.add(keras.layers.Flatten())
        
        # Dense FF ---------------------------------------------------------------------

    for i in range(0,hp['3dcnn_num_dense_layers'][h]):
        # Dense
        if hp['3dcnn_final_denselayer_size'][h][i] != []:
            nn.add(Dense(hp['3dcnn_final_denselayer_size'][h][i], 
                                      activation=hp['3dcnn_final_denselayer_activation'][h][i]))
        
        # Dropout
        if hp['3dcnn_final_dropout'][h][i] != []:
            nn.add(Dropout(hp['3dcnn_final_dropout'][h][i]))
        
    # The Final Activation Layer
    nn.add(Dense(1, activation='sigmoid'))

    return nn



def CNN_2D(hp,h):
    
    nn = Sequential()

    for i in range(0,hp['cnn2d_num_cnn_layers'][h]):
        #--------------------------------------------------------------------------
        # 3D Convolution
        if i == 0:
            nn.add(Conv2D(hp['cnn2d_filters_per_layer'][h][i],                        # 8
                                       kernel_size        = hp['cnn2d_kernal_sizes_per_layer'][h][i],     # (3, 3, 3),
                                       strides            = hp['cnn2d_strides_per_layer'][h][i],          # (1, 1, 1),
                                       kernel_initializer = hp['weights_initialization'][h],
                                       bias_initializer   = hp['bias_initialization'][h],
                                       input_shape        = hp['input_shape'][h], 
                                       data_format        ='channels_last',
                                       padding            ='same'))
        else:
            nn.add(Conv2D(hp['cnn2d_filters_per_layer'][h][i],                        # 8
                                       kernel_size        = hp['cnn2d_kernal_sizes_per_layer'][h][i],     # (3, 3, 3),
                                       strides            = hp['cnn2d_strides_per_layer'][h][i],          # (1, 1, 1),
                                       kernel_initializer = hp['weights_initialization'][h],
                                       bias_initializer   = hp['bias_initialization'][h],
                                       padding            = 'same'))

        
        #Batch Noramlization
        if hp['cnn2d_batchnormalize_per_layer'][h][i] != []:
            nn.add(BatchNormalization( axis            = hp['cnn2d_batchnormalize_axis'][h], 
                                momentum               = hp['cnn2d_batchnormalize_momentum'][h], 
                                epsilon                = hp['cnn2d_batchnormalize_epsilon'][h], 
                                center                 = hp['cnn2d_batchnormalize_center'][h], 
                                scale                  = hp['cnn2d_batchnormalize_scale'][h], 
                                beta_initializer       = hp['cnn2d_batchnormalize_beta_init'][h], 
                                gamma_initializer      = hp['cnn2d_batchnormalize_gamma_init'][h], 
                                moving_mean_initializer = hp['cnn2d_batchnormalize_moving_mean'][h], 
                                moving_variance_initializer =hp['cnn2d_batchnormalize_moving_var'][h], 
                                beta_regularizer       = None, 
                                gamma_regularizer      = None, 
                                beta_constraint        = None, 
                                gamma_constraint       = None))


        #Activation
        if hp['cnn2d_activations_per_layer'][h][i] != []:
            nn.add(Activation(hp['cnn2d_activations_per_layer'][h][i]))               # relu
        
        #3D Max pooling
        if hp['cnn2d_poolsize_per_layer'][h][i] != []:
            nn.add(MaxPooling2D(pool_size   = hp['cnn2d_poolsize_per_layer'][h][i],    # (3, 3, 3), 
                                padding     = 'same'))
        
        # Dropout
        if hp['cnn2d_dropouts_per_layer'][h][i] != []:
            nn.add(Dropout(hp['cnn2d_dropouts_per_layer'][h][i]))                            # 0.025


    #Flatten
    nn.add(keras.layers.Flatten())
        
        # Dense FF ---------------------------------------------------------------------

    for i in range(0,hp['cnn2d_num_dense_layers'][h]):
        # Dense
        if hp['cnn2d_final_denselayer_size'][h][i] != []:
            nn.add(Dense(hp['cnn2d_final_denselayer_size'][h][i], 
                                      activation=hp['cnn2d_final_denselayer_activation'][h][i]))
        
        # Dropout
        if hp['cnn2d_final_dropout'][h][i] != []:
            nn.add(Dropout(hp['cnn2d_final_dropout'][h][i]))
        
    # The Final Activation Layer
    nn.add(Dense(1, activation='sigmoid'))

    return nn


def callbacks(hp,h,fold_num):  
    # time_dir,checkpoint_period,early_stop_min_delta,early_stop_patience,reduce_lr_factor,reduce_lr_patience,reduce_lr_epsilon,reduce_lr_cooldown,reduce_lr_min_lr):  
    
    ###########################################################################
    # Checkpoints 
    ###########################################################################
    filepath="Checkpoints/" + hp['experiment'][h] + "/fold" + str(fold_num) + "/weights/nn_weights-{epoch:02d}.hdf5" # Where are checkpoints saved
    checkpoint = keras.callbacks.ModelCheckpoint(
                 filepath, 
                 monitor=['acc','loss','val_loss'],      # Validation set Loss           
                 verbose           = 0,                  # Display text 
                 save_weights_only = True,               # if True, only the model weights are saved
                 save_best_only    = False,              # if True, the latest-best model is overwritten
                 mode              = 'auto',             # used if 'save_best_only' is True  
                 period            = hp['checkpoint_period'][h]   # Epochs between checkpoints
                 )
    
    #TERMINATE ON NAN
    terminate_on_nan = keras.callbacks.TerminateOnNaN() #Terminates the run if nans occur
    
    #EARLY STOPPING IF THE VALIDATION LOSS STOPS IMPROVING
    early_stopping = keras.callbacks.EarlyStopping(
                     monitor   = 'val_loss',                      # criteria that is checked when we want to stop
                     min_delta = hp['early_stop_min_delta'][h],   # minimum amount things must change before we start applying the patience.
                     patience  = hp['early_stop_patience'][h],             # number of epochs with no improvement after which training will be stopped.
                     verbose   = 0,                               # output to console
                     mode      = 'auto')      
    
    #CHANGE THE LEARNING RATE
    learning_rate = keras.callbacks.ReduceLROnPlateau(
                    monitor    = 'val_loss',            # criteria that is checked when we want to stop
                    factor     = hp['reduce_lr_factor'][h],      # Reduce the loarning rate by a factor of 5 
                    patience   = hp['reduce_lr_patience'][h],    # If the performance platteaus for <n> iterations
                    verbose    = 0,                     # output to console
                    mode       = 'auto',                # direction that defines 'improvement' 
                    epsilon    =  hp['reduce_lr_epsilon'][h],     # minimum change
                    cooldown   =  hp['reduce_lr_cooldown'][h],    # then wait for 3 epochs before resuming normal activity 
                    min_lr     =  hp['reduce_lr_min_lr'][h])      # minimum learning rate
    
    #TENSOR BOARD - THE GRAPHICAL UI FOR REALITIME MONITORING OF PERFORMANCE
    #tensor_board = keras.callbacks.TensorBoard(
    #               log_dir          = "logs/{}".format(time()),
    #               histogram_freq   = 0,                    
    #               write_graph      = False, 
    #               write_grads      = False,
    #               write_images     = False)
    
    
    #CSV Logger - A log of the model performance.
    #log = keras.callbacks.CSVLogger('training.log', 
    #      separator  = ',', 
    #      append     = False)
    
    return [checkpoint,terminate_on_nan,early_stopping,learning_rate]



def gen_folders(hp,h,nn):                            
    os.mkdir('Checkpoints/' + hp['experiment'][h])                       # make the directory '<time_val>'
    os.mkdir('Checkpoints/' +  hp['experiment'][h] + '/topology')         # make the directory 'topology'
    os.mkdir('Checkpoints/' +  hp['experiment'][h] +'/all_performance')          # make the directory 'overall'
    
    for fold_num in range(0, hp['number_of_folds'][h]): 
        os.mkdir('Checkpoints/' +  hp['experiment'][h] + '/fold' + str(fold_num))                      # make the director for this fold
        os.mkdir('Checkpoints/' +  hp['experiment'][h] + '/fold' + str(fold_num) + '/weights')         # make the directory for weights
        os.mkdir('Checkpoints/' +  hp['experiment'][h] + '/fold' + str(fold_num) + '/performance')     # make the directory for performance

    json_nn = nn.to_json()
    with open('Checkpoints/' + hp['experiment'][h]  + '/topology/model.json', 'w') as outfile:
        json.dump(json_nn, outfile)
    

# GENERATES WELL-FORMATTED DATA
def gen_data(test_ind,files, outcomes, take_only):
    # INPUTS:
    #         files    : a python list with the name of all .npy files in your dataset.
    #         test_ind : an index that says which of the files, you want to combine
    #         outcomes : the binary outcome corresponding to each of the files
    #         take_only: the % of the total data from each file you want to take. e.g. .5 is 50%
    # OUTPUTS:
    #         Merged 'X' and 'Y' data
    ind = 0
    testing_files = files[test_ind]
    eg_data = np.load(testing_files[0].decode('utf8'))
    n_samples, n_frames, row, col, n_hertz = np.size(eg_data,0), np.size(eg_data,1), np.size(eg_data,2), np.size(eg_data,3), np.size(eg_data,4)

    #Initialize the data you want to return
    Y_test = np.float16(np.zeros(round(n_samples*np.size(test_ind)*take_only)))
    X_test = np.float16(np.zeros([round(n_samples*np.size(test_ind)*take_only),
                       n_frames,
                       row,
                       col,
                       n_hertz]))
    
    # append the data together.
    for i in range(0,np.size(testing_files)):
        this_data = np.load(testing_files[i].decode('utf8'))
        X_test[ind:int(ind+round(n_samples*take_only)),:,:,:,:] = np.float16(this_data[0:round(np.size(this_data,0)*take_only),:,:,:,:])
        Y_test[ind:int(ind+round(n_samples*take_only))]         = np.float16(outcomes[i]*np.ones(round(np.size(this_data,0)*take_only)))
        ind = int(ind+round(n_samples*take_only)) 
    
    return X_test, Y_test



#GENERATES TRAIN, TEST, AND VALIDATION FOR A SIMPLE TABLE DATASET
def split_bin_data(data,outcome,train_perc,val_perc,test_perc):
    # Split the data into training training, testing, and validation                   
    [X_train, X_test, Y_train, Y_test] = train_test_split(data,                    # The training dataset, first dimention is where the split occurs
                                                          outcome,                 # The outcome data (1/0)
                                                          test_size=1-train_perc   # The % of the data for testing
                                                          )
      
    [X_val, X_test, Y_val, Y_test] = train_test_split(X_test,                      
                                                      Y_test,                     
                                                      test_size=test_perc/(test_perc+val_perc)  
                                                      )
    
    return [X_train, X_val, X_test, Y_train, Y_val, Y_test]



def compile_cross_fold_results(hp,h,epochs_completed):
    ###############################################################################
    # COLLECT ALL PERFORMANCE AND STORE IN ALL 'all_performance/'                                      
    ############################################################################### 
    all_train  = np.zeros([hp['number_of_folds'][h],epochs_completed-1,34])
    all_val    = np.zeros([hp['number_of_folds'][h],epochs_completed-1,34])
    all_test   = np.zeros([hp['number_of_folds'][h],epochs_completed-1,34])
    
    for i in range(0,hp['number_of_folds'][h]):
        all_train[i,:,:] = np.load('Checkpoints/' + hp['experiment'][h] + '/fold' + str(i) +'/performance/train_performance.npy')
        all_val[i,:,:]   = np.load('Checkpoints/' + hp['experiment'][h] + '/fold' + str(i) +'/performance/val_performance.npy')
        all_test[i,:,:]  = np.load('Checkpoints/' + hp['experiment'][h] + '/fold' + str(i) +'/performance/test_performance.npy')
    
    np.save('Checkpoints/' + hp['experiment'][h] + '/all_performance/all_train',all_train)
    np.save('Checkpoints/' + hp['experiment'][h] + '/all_performance/all_val',all_val)
    np.save('Checkpoints/' + hp['experiment'][h] + '/all_performance/all_test',all_test)


# EVALUATES THE MODEL ACCORDING TO SEVERAL PERFORMANCE METRICS.

 
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

#---------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.50)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

#----------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.50)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score

# FROM https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41108
def jacek_auc(y_true, y_pred):
   score, up_opt = tf.metrics.auc(y_true, y_pred)
   #score, up_opt = tf.contrib.metrics.streaming_auc(y_pred, y_true)    
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score

# FROM https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41015
# AUC for a binary classifier
def discussion41015_auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)


def train_NN(hp,h,fold_num,nn,X_train,Y_train,X_val,Y_val):

    # Generate weights for the training data
    if hp['balance_class_cost'][h] == True:
        class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    elif hp['balance_class_cost'][h] == False:
        if type(hp['class_weights'][h]) == list:
         class_weights = {0 : hp['class_weights'][h][0],
                          1 : hp['class_weights'][h][1]} 
    
    ###############################################################################
    # USE PARAMETERS TO INITIALIZE OPTIMIZATION ALGORITHM              
    ###############################################################################       
    # Optimizer Selection using parameters
    grad_desc_algorithm = choose_optimizer(hp,h)  
    
    # Specfify model callbacks using parameters
    [checkpoint,terminate_on_nan,early_stopping,learning_rate] =  callbacks(hp,h,fold_num)  
    
    # compile the network
    nn.compile(loss = hp['loss_function'][h], 
                      optimizer = grad_desc_algorithm, 
                      metrics = ['accuracy'])
    
    
    #Determine the batch size for NN training
    batch_size = int(round(hp['batch_perc'][h]*np.size(X_train,0)))
    
    ###############################################################################
    # FIT THE MODELS
    ###############################################################################
    # Fit the model: https://keras.io/models/model/#fit
    fit_nn = nn.fit(X_train,        # Training Data X
                    Y_train,                          # Training Data Y
                    validation_data = (X_val,
                                       Y_val),  # Validation data tuple
                    shuffle         = 1,              # shuffle the training data epoch before you use it
                    initial_epoch   = 0,              # Starting Epoch (should awlways be 0)
                    epochs          = hp['num_epochs'][h],     # Number of runs through the data 
                    batch_size      = batch_size,     # Number of samples per gradient update. 
                    verbose         = 1,              # display options to console
                    class_weight    = class_weights,  # Force balanced class penalties 
                    callbacks=[checkpoint])            # We want to save checkpoints
                               #terminate_on_nan,      # terminate run if nans are found                      
                               #learning_rate,         # How to change learning rate.
                               #early_stopping,        # The criteria for early stopping 
                  


    return fit_nn    

def choose_optimizer(hp,h):
    if hp['optimizer'][h] == 'sgd':
        grad_desc_algorithm = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)     
        if type(hp['lr'][h])       != list: 
            grad_desc_algorithm.lr = hp['lr'][h]
        if type(hp['decay'][h])    != list: 
            grad_desc_algorithm.decay = hp['decay'][h]
        if type(hp['momentum'][h]) != list: 
            grad_desc_algorithm.momentum = hp['momentum'][h]
        if type(hp['nesterov'][h]) != list:
            grad_desc_algorithm.nesterov = hp['nesterov'][h]  
    elif hp['optimizer'][h] == 'rmsprop':    
        grad_desc_algorithm = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=K.epsilon(), decay=0.0)
        if type(hp['lr'][h])      != list:
             grad_desc_algorithm.lr = hp['lr'][h]
        if type(hp['rho'][h])     != list:
             grad_desc_algorithm.rho = hp['rho'][h]
        if type(hp['epsilon'][h]) != list:
            grad_desc_algorithm.epsilon = hp['epsilon'][h]
        if type(hp['decay'][h])   != list:
             grad_desc_algorithm.decay = hp['decay'][h]
    elif hp['optimizer'][h] == 'adagrad':    
        grad_desc_algorithm = keras.optimizers.Adagrad(lr=0.01, epsilon=K.epsilon(), decay=0.0)
        if type(hp['lr'][h])      != list:
             grad_desc_algorithm.lr = hp['lr'][h]
        if type(hp['epsilon'][h]) != list:
            grad_desc_algorithm.epsilon = hp['epsilon'][h]
        if type(hp['decay'][h])   != list:
             grad_desc_algorithm.decay = hp['decay'][h]        
    elif hp['optimizer'][h] == 'adadelta':
        grad_desc_algorithm = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=K.epsilon(), decay=0.0)
        if type(hp['lr'][h])      != list:
             grad_desc_algorithm.lr = hp['lr'][h]
        if type(hp['rho'][h])     != list:
             grad_desc_algorithm.rho = hp['rho'][h]
        if type(hp['epsilon'][h]) != list:
            grad_desc_algorithm.epsilon = hp['epsilon'][h]
        if type(hp['decay'][h])   != list:
             grad_desc_algorithm.decay = hp['decay'][h]       
    elif hp['optimizer'][h] == 'adam':     
        grad_desc_algorithm = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=K.epsilon(), decay=0.0)
        if type(hp['lr'][h])      != list:
             grad_desc_algorithm.lr = hp['lr'][h]
        if type(hp['beta_1'][h])  != list:
             grad_desc_algorithm.beta_1 = hp['beta_1'][h]
        if type(hp['beta_2'][h])  != list:
            grad_desc_algorithm.beta_2 = hp['beta_2'][h]
        if type(hp['epsilon'][h]) != list:
             grad_desc_algorithm.epsilon = hp['epsilon'][h]
        if type(hp['decay'][h])   != list:
             grad_desc_algorithm.decay = hp['decay'][h]         
    elif hp['optimizer'][h] == 'adamax':
        grad_desc_algorithm = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=K.epsilon(), decay=0.0)
        if type(hp['lr'][h])      != list:
             grad_desc_algorithm.lr = hp['lr'][h]
        if type(hp['beta_1'][h])  != list:
             grad_desc_algorithm.beta_1 = hp['beta_1'][h]
        if type(hp['beta_2'][h])  != list:
            grad_desc_algorithm.beta_2 = hp['beta_2'][h]
        if type(hp['epsilon'][h]) != list:
             grad_desc_algorithm.epsilon = hp['epsilon'][h]
        if type(hp['decay'][h])   != list:
             grad_desc_algorithm.decay = hp['decay'][h]       
    elif hp['optimizer'][h] == 'nadam':
        grad_desc_algorithm = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=K.epsilon(), schedule_decay=0.004)
        if type(hp['lr'][h])      != list:
             grad_desc_algorithm.lr = hp['lr'][h]
        if type(hp['beta_1'][h])  != list:
             grad_desc_algorithm.beta_1 = hp['beta_1'][h]
        if type(hp['beta_2'][h])  != list:
            grad_desc_algorithm.beta_2 = hp['beta_2'][h]
        if type(hp['epsilon'][h]) != list:
             grad_desc_algorithm.epsilon = hp['epsilon'][h]
        if type(hp['decay'][h])   != list:
             grad_desc_algorithm.schedule_decay = hp['decay'][h] 
    
    if type(hp['clipnorm'][h])  != list:
        grad_desc_algorithm.clipnorm = hp['clipnorm'][h]
    if type(hp['clipvalue'][h])  != list:
        grad_desc_algorithm.clipnorm = hp['clipvalue'][h]      
    
    return grad_desc_algorithm


def draw_hp_tuples_desc(p,layers):
    g = []
    for i in range(0,layers):
        if i == 0:
            g.append(p[np.random.randint(np.size(p,0))])
        elif i > 0:
            d =p[np.random.randint(np.size(p,0))]
            while d[0] > g[-1][0]:
                d = p[np.random.randint(np.size(p,0))]
            g.append(d)
    return g



def draw_hp_tuples_desc_with_missing(p,layers):
    g = []
    for i in range(0,layers):
        if i == 0:
            g.append(p[np.random.randint(len(p))])
        elif i > 0:
            d = p[np.random.randint(len(p))]
            if d != [] and g[-1] != []:
                while d == [] or d[0] > g[-1][0]:
                    d = p[np.random.randint(len(p))]
                g.append(d)
            else:
                g.append(d)
    return g


def draw_hp_with_missing(p,layers):
    g = []
    for i in range(0,layers):
        g.append(p[np.random.randint(len(p))])     
    return g


def draw_hp_contiguous_asc(p,starting_from,layers):
    g = []
    for i in range(0,layers):
        if i == 0:
            ind = np.random.randint(starting_from) 
            g.append(p[ind])
        elif i > 0:
            ind = ind + 1
            g.append(p[ind])
    return g


###############################################################################
# PLOT PERFORMANCE OVER TIME ...                                              
################################################################################ 
#    xdim = 5
#    ydim = 7
#    
#    # generate the plots
#    fig,axarr = plt.subplots(xdim, ydim,figsize=(16, 18), dpi=100)
#    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9 , top=0.9, wspace=0.4, hspace=0.4)
#    
#    ind = 0
#    for i in range(0,xdim):
#        for j in range(0,ydim):
#            if ind < np.size(train_over_time,1):
#                #Plot the training
#                axarr[i,j].plot(range(checkpoint_period,epochs_completed,checkpoint_period), 
#                                train_over_time[:,ind],
#                                linestyle='-', 
#                                color='blue',
#                                label='Training', 
#                                lw=2)
#                #Plot the validation
#                axarr[i,j].plot(range(checkpoint_period,epochs_completed,checkpoint_period), 
#                                val_over_time[:,ind],
#                                linestyle='-', 
#                                color='red',
#                                label='Validation', 
#                                lw=2)
#                #Plot the testing 
#                axarr[i,j].plot(range(checkpoint_period,epochs_completed,checkpoint_period), 
#                                test_over_time[:,ind],
#                                linestyle='-', 
#                                color='green',
#                                label='Test', 
#                                lw=2)
#                
#                #Title the plots according to the names
#                axarr[i,j].set_title(test_names[ind], fontsize=8)
#                
#                #Strip off the boxes
#                axarr[i,j].spines['top'].set_visible(False)
#                axarr[i,j].spines['right'].set_visible(False)
#                axarr[i,j].spines['bottom'].set_visible(False)
#                axarr[i,j].spines['left'].set_visible(False)
#                
#                #If this is the last row of plots
#                if ind >= (xdim*ydim)-ydim:
#                    axarr[i,j].set_xlabel('Training Epoch')
#                else:
#                    axarr[i,j].set_xticks([])
#                    axarr[i,j].set_xticklabels([])
#            
#            #If we've run out of data clear the plots
#            elif ind >= np.size(train_over_time,1):
#                leg = axarr[i,j-1].legend(bbox_to_anchor=(1.3, .75), loc=2, borderaxespad=0.,fontsize=13)
#                axarr[i,j].set_xticklabels([])
#                axarr[i,j].set_xticks([])
#                axarr[i,j].set_visible(False)
#            # Stick the legend in the final plot
#            
#            ind = ind +1
#    
#    plt.savefig('Checkpoints/' + time_dir + '/fold' + str(fold_num) +'/performance/perf.png')
