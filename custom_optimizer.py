#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:10:05 2018

@author: mohammad
"""
import keras
import keras.backend as K
def choose_optimizer(optimizer,lr,decay,momentum,rho,epsilon,beta_1,beta_2,nesterov,amsgrad):
    if optimizer == 'sgd':
        grad_desc_algorithm = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        if type(lr)       != list: 
            grad_desc_algorithm.lr = lr
        if type(decay)    != list: 
            grad_desc_algorithm.decay = decay
        if type(momentum) != list: 
            grad_desc_algorithm.momentum = momentum
        if type(nesterov) != list:
            grad_desc_algorithm.nesterov = nesterov   
    elif optimizer == 'rmsprop':    
        grad_desc_algorithm = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=K.epsilon(), decay=0.0)
        if type(lr)      != list:
             grad_desc_algorithm.lr = lr
        if type(rho)     != list:
             grad_desc_algorithm.rho = rho
        if type(epsilon) != list:
            grad_desc_algorithm.epsilon = epsilon
        if type(decay)   != list:
             grad_desc_algorithm.decay = decay
    elif optimizer == 'adagrad':    
        grad_desc_algorithm = keras.optimizers.Adagrad(lr=0.01, epsilon=K.epsilon(), decay=0.0)
        if type(lr)      != list:
             grad_desc_algorithm.lr = lr
        if type(epsilon) != list:
            grad_desc_algorithm.epsilon = epsilon
        if type(decay)   != list:
             grad_desc_algorithm.decay = decay        
    elif optimizer == 'adadelta':
        grad_desc_algorithm = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=K.epsilon(), decay=0.0)
        if type(lr)      != list:
             grad_desc_algorithm.lr = lr
        if type(rho)     != list:
             grad_desc_algorithm.rho = rho
        if type(epsilon) != list:
            grad_desc_algorithm.epsilon = epsilon
        if type(decay)   != list:
             grad_desc_algorithm.decay = decay       
    elif optimizer == 'adam':     
        grad_desc_algorithm = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=K.epsilon(), decay=0.0)
        if type(lr)      != list:
             grad_desc_algorithm.lr = lr
        if type(beta_1)  != list:
             grad_desc_algorithm.beta_1 = beta_1
        if type(beta_2)  != list:
            grad_desc_algorithm.beta_2 = beta_2
        if type(epsilon) != list:
             grad_desc_algorithm.epsilon = epsilon
        if type(decay)   != list:
             grad_desc_algorithm.decay = decay         
    elif optimizer == 'adamax':
        grad_desc_algorithm = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=K.epsilon(), decay=0.0)
        if type(lr)      != list:
             grad_desc_algorithm.lr = lr
        if type(beta_1)  != list:
             grad_desc_algorithm.beta_1 = beta_1
        if type(beta_2)  != list:
            grad_desc_algorithm.beta_2 = beta_2
        if type(epsilon) != list:
             grad_desc_algorithm.epsilon = epsilon
        if type(decay)   != list:
             grad_desc_algorithm.decay = decay       
    elif optimizer == 'nadam':
        grad_desc_algorithm = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=K.epsilon(), schedule_decay=0.004)
        if type(lr)      != list:
             grad_desc_algorithm.lr = lr
        if type(beta_1)  != list:
             grad_desc_algorithm.beta_1 = beta_1
        if type(beta_2)  != list:
            grad_desc_algorithm.beta_2 = beta_2
        if type(epsilon) != list:
             grad_desc_algorithm.epsilon = epsilon
        if type(decay)   != list:
             grad_desc_algorithm.schedule_decay = decay 
    
    return grad_desc_algorithm
