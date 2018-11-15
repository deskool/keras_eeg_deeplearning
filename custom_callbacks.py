#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:20:24 2018

@author: Mohammad Ghassemi, PhD Candidate, MIT
"""

#MODEL CHECKPOINTS
import keras
from keras.callbacks import ModelCheckpoint
from time import time

def callbacks(time_dir,checkpoint_period,early_stop_min_delta,early_stop_patience,reduce_lr_factor,reduce_lr_patience,reduce_lr_epsilon,reduce_lr_cooldown,reduce_lr_min_lr):  
    ###########################################################################
    # Checkpoints 
    ###########################################################################
    filepath="Checkpoints/" + time_dir + "/weights/nn_weights-{epoch:02d}.hdf5" # Where are checkpoints saved
    checkpoint = keras.callbacks.ModelCheckpoint(
                 filepath, 
                 monitor=['acc','loss','val_loss'],      # Validation set Loss           
                 verbose           = 0,                  # Display text 
                 save_weights_only = True,               # if True, only the model weights are saved
                 save_best_only    = False,              # if True, the latest-best model is overwritten
                 mode              = 'auto',             # used if 'save_best_only' is True  
                 period            = checkpoint_period   # Epochs between checkpoints
                 )
    
    #TERMINATE ON NAN
    terminate_on_nan = keras.callbacks.TerminateOnNaN() #Terminates the run if nans occur
    
    #EARLY STOPPING IF THE VALIDATION LOSS STOPS IMPROVING
    early_stopping = keras.callbacks.EarlyStopping(
                     monitor   = 'val_loss',             # criteria that is checked when we want to stop
                     min_delta = early_stop_min_delta,   # minimum amount things must change before we start applying the patience.
                     patience  = early_stop_patience,    # number of epochs with no improvement after which training will be stopped.
                     verbose   = 0,                      # output to console
                     mode      = 'auto')      
    
    #CHANGE THE LEARNING RATE
    learning_rate = keras.callbacks.ReduceLROnPlateau(
                    monitor    = 'val_loss',            # criteria that is checked when we want to stop
                    factor     = reduce_lr_factor,      # Reduce the loarning rate by a factor of 5 
                    patience   = reduce_lr_patience,    # If the performance platteaus for <n> iterations
                    verbose    = 0,                     # output to console
                    mode       = 'auto',                # direction that defines 'improvement' 
                    epsilon    = reduce_lr_epsilon,     # minimum change
                    cooldown   = reduce_lr_cooldown,    # then wait for 3 epochs before resuming normal activity 
                    min_lr     = reduce_lr_min_lr)      # minimum learning rate
    
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
