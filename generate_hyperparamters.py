#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:24:06 2018

@author: mohammad
"""
import numpy as np
import custom                                                                  # Import our custom functions that make this file clean.
from six.moves import reload_module
reload_module(custom)  

experiment = []
plot_perf = []
take_only = []
input_shape = []
seed =  []
train_perc = []
val_perc = []
test_perc = []
checkpoint_period = []
number_of_folds = []
number_of_folds = []
number_of_epochs = []
batch_perc = []
balance_class_cost = []
class_weights = []
weights_initialization = []
bias_initialization = []
loss_function = []
optimizer = []
optimizer = []
lr = []#                               : [[],[]],                              
clipnorm = []#                         : [1,[]],
clipvalue = []#                        : [0.5,[]],
decay = []#                            : [[],[]],                             
momentum = []#                         : [[],[]],                             
rho = []#                              : [[],[]],                             # 
epsilon = []#                          : [[],[]],                             # 
beta_1 = []#                           : [[],[]],                             # 
beta_2 = []#                           : [[],[]],                             # 
nesterov = []#                         : [[],[]],                             # 
amsgrad = []#                          : [[],[]],                             #                              
early_stop_min_delta = []#             : [0,0],                               # 
early_stop_patience = []#              : [100,100],                           # 
reduce_lr_patience = []#               : [5,5],                                
reduce_lr_factor = []#                 : [0.2,0.2],                           # 
reduce_lr_epsilon = []#                : [0.0001,0.0001],                     
reduce_lr_cooldown = []#               : [3,3],                               
reduce_lr_min_lr = []#                 : [0.001,0.001],                       
use_CNN_3D = []#                       : [0 , 1],                             
cnn3d_num_layers = []#                 : [[], 2],                             
cnn3d_filters_per_layer = []#          : [[], [8,16]],                        
cnn3d_kernal_sizes_per_layer = []#     : [[], [(3,3,3),(2,2,2)]],             
cnn3d_strides_per_layer = []#          : [[], [(2,2,2),(1,1,1)]],             
cnn3d_batchnormalize_per_layer = []#   : [[], [1,1]],                         
cnn3d_batchnormalize_axis = []#        : [[], -1],                            
cnn3d_batchnormalize_momentum = []#    : [[], 0.99],                          
cnn3d_batchnormalize_epsilon = []#     : [[], 0.001],                         
cnn3d_batchnormalize_center = []#      : [[], True],                          
cnn3d_batchnormalize_scale = []#       : [[], True],                          
cnn3d_batchnormalize_beta_init = []#   : [[], 'zeros'],                       
cnn3d_batchnormalize_gamma_init = []#  : [[], 'ones'],                        
cnn3d_batchnormalize_moving_mean = []# : [[], 'zeros'],                       
cnn3d_batchnormalize_moving_var = []#  : [[], 'ones'],                        
cnn3d_activations_per_layer = []#      : [[], ['relu','relu']],               
cnn3d_poolsize_per_layer = []#         : [[], [(3,3,3),(3,3,3)]],             
cnn3d_dropouts_per_layer = []#         : [[], [.25, [] ]],                    
cnn3d_num_dense_layers = []#           : [[], 2],                             
cnn3d_final_denselayer_size = []#      : [[], [64,64]],                       
cnn3d_final_denselayer_activation = []#: [[], ['sigmoid','sigmoid']],         
cnn3d_final_dropout = []#              : [[], [.25,.25]],                      
use_LRCN = []#                         : [0,0],                                       
lrcn_num_cnn_layers = []#              : [2,[]],                              
lrcn_cnn_filters_per_layer = []#       : [[8,8], [] ],                        
lrcn_cnn_strides_per_layer = []#       : [[(1,1),(1,1)], [] ],                
lrcn_cnn_kernal_size_per_layer = []#   : [[(2,2),(2,2)], [] ],                
lrcn_batchnormalize_per_layer = []#    : [[1,1],[]],                          
lrcn_batchnormalize_axis = []#         : [-1,[]],                             
lrcn_batchnormalize_momentum = []#     : [0.99,[]],                           
lrcn_batchnormalize_epsilon = []#      : [0.001,[]],                          
lrcn_batchnormalize_center = []#       : [True,[]],                           
lrcn_batchnormalize_scale = []#        : [True,[]],                           
lrcn_batchnormalize_beta_init = []#    : ['zeros',[]],                        
lrcn_batchnormalize_gamma_init = []#   : ['ones',[]],                         
lrcn_batchnormalize_moving_mean = []#  : ['zeros',[]],                        
lrcn_batchnormalize_moving_var = []#   : ['ones',[]],                         
lrcn_activation_type_per_layer = []#   : [['relu','relu'], [] ],              
lrcn_cnn_poolsize_per_layer = []#      : [[  [] ,(2,2)], [] ],                
lrcn_cnn_poolstrides_per_layer = []#   : [[  [] ,(1,1)], [] ],                
lrcn_dropouts_per_layer = []#          : [[ .25 , []  ], [] ],                
lrcn_pre_lstm_dropout = []#            : [0.5, [] ],                          
lrcn_lstm_dropout = []#                : [0.0, [] ],                        
lrcn_lstm_recurrent_dropout = []#      : [0.01,[] ],                           
lrcn_lstm_output_dim = []#             : [4, [] ],                            
lrcn_lstm_activation = []#             : ['tanh', [] ],                       
lrcn_lstm_recurrent_activation = []#   : ['hard_sigmoid', [] ],               
lrcn_lstm_recurrent_initializer = []#  : ['orthogonal', [] ],                 
lrcn_lstm_use_bias = []#               : [ True, []],                         
lrcn_ltsm_kernal_initializer = []#     : ['glorot_uniform', []],              
lrcn_lstm_bias_intitializer = []#      : ['zeros',[]],                         
use_CNN_2D = []#                       : [1,0],                               
cnn2d_num_cnn_layers = []#             : [2,[]],                              
cnn2d_filters_per_layer = []#          : [[32,64],[]],                        
cnn2d_kernal_sizes_per_layer = []#     : [[(3,3),(2,2)],[]],      
cnn2d_strides_per_layer = []#          : [[(1,1),(1,1)],[]],      
cnn2d_batchnormalize_per_layer = []#   : [[[],[]],[]],                     
cnn2d_batchnormalize_axis = []#        : [-1,[]],                             
cnn2d_batchnormalize_momentum = []#    : [0.99,[]],                           
cnn2d_batchnormalize_epsilon = []#     : [0.001,[]],                          
cnn2d_batchnormalize_center = []#      : [True,[]],                           
cnn2d_batchnormalize_scale = []#       : [True,[]],                           
cnn2d_batchnormalize_beta_init = []#   : ['zeros',[]],                        
cnn2d_batchnormalize_gamma_init = []#  : ['ones',[]],                         
cnn2d_batchnormalize_moving_mean = []# : ['zeros',[]],                        
cnn2d_batchnormalize_moving_var = []#  : ['ones',[]],                         
cnn2d_activations_per_layer = []#      : [['relu','relu'],[]],  
cnn2d_poolsize_per_layer = []#         : [[[],(2,2)],[]],      
cnn2d_dropouts_per_layer = []#         : [[[], .25],[]],               
cnn2d_num_dense_layers = []#           : [1,[]],                              
cnn2d_final_denselayer_size = []#      : [[128],[]],                           
cnn2d_final_denselayer_activation = []                    
cnn2d_final_dropout = []                       

np.random.seed(1337)

for a in range(0,300):

    #-------------------------------------------------------------------------------------------------------
    # Experiment Initialization
    #------------------------------------------------------------------------------------------------------- 
    # string, Name of result directory set to "str(time.time())", for multiple runs 
    if a % 3 == 0:
        experiment.append('2DCNN_' + str(a)) 
    
        # tuple, the shape of the input data
        input_shape.append((8,8,4))
    
        # bool    , Set to 1 if you want to use this model, for this experiment
        p = [1]
        use_CNN_2D.append((p[np.random.randint(len(p))]))
    
        # int      , Set to 1 if you want to use this model
        p = [0]
        use_CNN_3D.append((p[np.random.randint(len(p))]))
        
        # int     , Set to 1 if you want to use this model for this experiment 
        p = [0]
        use_LRCN.append((p[np.random.randint(len(p))]))    
    
    if a % 3 == 1:
        experiment.append('LRCN_' + str(a)) 
    
        # tuple, the shape of the input data
        input_shape.append((72,8,8,4))
    
        # bool    , Set to 1 if you want to use this model, for this experiment
        p = [0]
        use_CNN_2D.append((p[np.random.randint(len(p))]))
    
        # int      , Set to 1 if you want to use this model
        p = [0]
        use_CNN_3D.append((p[np.random.randint(len(p))]))
        
        # int     , Set to 1 if you want to use this model for this experiment 
        p = [1]
        use_LRCN.append((p[np.random.randint(len(p))]))    
    if  a % 3 == 2:        
        experiment.append('3DCNN_' + str(a)) 
    
        # tuple, the shape of the input data
        input_shape.append((72,8,8,4))
    
        # bool    , Set to 1 if you want to use this model, for this experiment
        p = [0]
        use_CNN_2D.append((p[np.random.randint(len(p))]))
    
        # int      , Set to 1 if you want to use this model
        p = [1]
        use_CNN_3D.append((p[np.random.randint(len(p))]))
        
        # int     , Set to 1 if you want to use this model for this experiment 
        p = [0]
        use_LRCN.append((p[np.random.randint(len(p))]))    
        
    # int   , Generate a plot of the performanc over time.  
    plot_perf.append(0)
    
    # float, % of the total data you want to keep
    take_only.append(.0016666666666666668*100)
    
    # int  , Random Seed 
    seed.append(1) 
    
    # flaot, Training percentage
    train_perc.append(.8)
    
    # float, Validation percentage
    val_perc.append(.2)
    
    # float, Test percentage
    test_perc.append(0)
    
    # int  , >= 1 , Epochs before we saving our weights
    checkpoint_period.append(5)    
    
    # int  , >= 1 , Number of Cross validation folds
    number_of_folds.append(5)
    
    # int  , >= 1 , Number of training data epochs 
    number_of_epochs.append(125)
    
    # float, Percentage of training to use data per training batch (1 = 100%) 
    batch_perc.append(.01)
    
    # bool , T|F  , weights data balance misclassificaiton costs if True
    balance_class_cost.append(False)
    
    # [float], The weights of classes 0 and 1  
    class_weights.append([.5,.5])
    
    #-------------------------------------------------------------------------------------------------------
    # Weights Initialization
    #------------------------------------------------------------------------------------------------------- 
    p = ['lecun_normal','he_normal','glorot_normal']
    weights_initialization.append(p[np.random.randint(len(p))])

    #Initialization of the bias
    p = ['Zeros']
    bias_initialization.append(p[np.random.randint(len(p))])

    #-------------------------------------------------------------------------------------------------------
    # Loss Function
    #-------------------------------------------------------------------------------------------------------
    p = ['binary_crossentropy']
    loss_function.append(p[np.random.randint(len(p))])  

    #-------------------------------------------------------------------------------------------------------
    # Optimization Paramters
    #-------------------------------------------------------------------------------------------------------
	
    #Optimizer: sgd|rmsprop|adagrad|adadelta|adam|adamaz|nadam 
    p = ['sgd']
    optimizer.append(p[np.random.randint(len(p))])   
    
    #float, >= 0, the learning rate of the optimizer  
    p = [0.01, 1e-02,1e-03,1e-04,1e-05,1e-06,1e-07,1e-08,1e-09]
    lr.append((p[np.random.randint(len(p))]))  

    p = [1]
    clipnorm.append((p[np.random.randint(len(p))]))  

    p = [0.5]
    clipvalue.append((p[np.random.randint(len(p))]))

    # float, >= 0, the decay in the learning rate, per epoch
    p = [[]]
    decay.append((p[np.random.randint(len(p))]))

    # float, >= 0, momentum parameter during sgd
    p = [[]]
    momentum.append((p[np.random.randint(len(p))]))

    # float, >= 0, rho
    p = [[]]
    rho.append((p[np.random.randint(len(p))]))

    # float, >= 0, epsilon
    p = [[]]
    epsilon.append((p[np.random.randint(len(p))]))

    # float, 0 < beta < 1
    p = [[]]
    beta_1.append((p[np.random.randint(len(p))]))

    # float, 0 < beta < 1
    p = [[]]
    beta_2.append((p[np.random.randint(len(p))]))

    # bool , T|F
    p = [[]]
    nesterov.append((p[np.random.randint(len(p))]))

    # bool , T|F
    p = [[]]
    amsgrad.append((p[np.random.randint(len(p))]))

    #float, minimum amount things must change before we start applying the patience.
    p = [0]
    early_stop_min_delta.append((p[np.random.randint(len(p))]))

    #int  ,  number of epochs with no improvement after which training will be stopped.
    p = [100]
    early_stop_patience.append((p[np.random.randint(len(p))]))

    # float, How long must performance platteaus before lr is reduced 
    p = [5]
    reduce_lr_patience.append((p[np.random.randint(len(p))]))

    # float, Amount to reduce the loarning rate by: .2 is a factor of 5 
    p = [0.2]
    reduce_lr_factor.append((p[np.random.randint(len(p))]))

    # float, threshold for measuring new optimum
    p = [0.0001]
    reduce_lr_epsilon.append((p[np.random.randint(len(p))]))

    # int  , Number of epochs to keep at this lr before resuming normal activity
    p = [3]
    reduce_lr_cooldown.append((p[np.random.randint(len(p))]))

    # float, minimum learning rate we can reduce to
    p = [0.001]
    reduce_lr_min_lr.append((p[np.random.randint(len(p))]))

    #-------------------------------------------------------------------------------------------------------
    # 3D CNN 
    #-------------------------------------------------------------------------------------------------------
    # int      . Number of Convolational layers in the 3D CNN
    p = [4,3,2,1]
    cnn3d_num_layers.append(p[np.random.randint(len(p))])

    # [int]    , per layer -Number of CNN filters
    p = [4,8,16,32,64,128]
    g = custom.draw_hp_contiguous_asc(p,3,cnn3d_num_layers[-1])
    cnn3d_filters_per_layer.append(g)

    # [(int)]  , per layer - CNN Kernal sizes
    p = [(4,4,4),(3,3,3),(2,2,2)]
    g = custom.draw_hp_tuples_desc(p,cnn3d_num_layers[-1])
    cnn3d_kernal_sizes_per_layer.append(g)

    # [(int)]  , per layer - CNN Strides
    p = [(2,2,2),(1,1,1),(1,1,1)]
    g = custom.draw_hp_tuples_desc(p,cnn3d_num_layers[-1])
    cnn3d_strides_per_layer.append(g)    

    # [int]   , per-layer, do you want to batch normalize
    p = [1,0]
    g = list(np.repeat(p[np.random.randint(len(p))],cnn3d_num_layers[-1]))
    cnn3d_batchnormalize_per_layer.append(g)
    
    # int     , axis to perform batch normalization
    p = [-1]
    cnn3d_batchnormalize_axis.append((p[np.random.randint(len(p))]))

    # float   , momentum for the moving mean and moviing variance
    p = [0.99]
    cnn3d_batchnormalize_momentum.append((p[np.random.randint(len(p))]))

    # float   , small float aadded to variance to avoid dividing by 0    
    p = [0.001]
    cnn3d_batchnormalize_epsilon.append((p[np.random.randint(len(p))]))

    # Bool    , if True, add offset of beta to normalized tensor
    p = [True]
    cnn3d_batchnormalize_center.append((p[np.random.randint(len(p))]))
    
    # Bool    , if True, multiply by gamma, if false, gamma is not used
    p = [True]
    cnn3d_batchnormalize_scale.append((p[np.random.randint(len(p))]))

    # string  , initializer for the beta weights
    p = ['zeros']
    cnn3d_batchnormalize_beta_init.append((p[np.random.randint(len(p))]))

    # string  , initializer for the gamma weigths
    p = ['ones']
    cnn3d_batchnormalize_gamma_init.append((p[np.random.randint(len(p))]))

    # string  , initializer for the moving mean
    p = ['zeros']
    cnn3d_batchnormalize_moving_mean.append((p[np.random.randint(len(p))]))

    # string  , initializer for the moving variance
    p = ['ones']
    cnn3d_batchnormalize_moving_var.append((p[np.random.randint(len(p))]))

    # [string] , per layer - CNN activation
    p = ['relu']
    g = list(np.repeat(p[np.random.randint(len(p))],cnn3d_num_layers[-1]))
    cnn3d_activations_per_layer.append(g)

    # [(int)]  , per layer - CNN maxpooling pool-size
    p = [(2,2,2),[]]
    g = custom.draw_hp_tuples_desc_with_missing(p,cnn3d_num_layers[-1])
    cnn3d_poolsize_per_layer.append(g)

    # [float]  , per layer -  CNN dropouts
    p = [.1,.2,.3,.4,.5,.6,[]]
    g = custom.draw_hp_with_missing(p,cnn3d_num_layers[-1])
    cnn3d_dropouts_per_layer.append(g)

    # int      , Number of dense layers at the very end
    p = [2,1]
    cnn3d_num_dense_layers.append((p[np.random.randint(len(p))]))

    # [int]    , per layer - Number of nodes in the final dense layer
    p = [5, 10, 20, 40, 60, 120]
    g = custom.draw_hp_with_missing(p,cnn3d_num_dense_layers[-1])
    cnn3d_final_denselayer_size.append(g)

    # [string] , per layer - Activation fucntion of the final layer
    p = ['relu','tanh']
    g = custom.draw_hp_with_missing(p,cnn3d_num_dense_layers[-1])
    cnn3d_final_denselayer_activation.append(g)

    # [float]  , per layer - Dropout of the final layer  
    p = [.1,.2,.3,.4,.5,.6,[]]
    g = custom.draw_hp_with_missing(p,cnn3d_num_dense_layers[-1])
    cnn3d_final_dropout.append(g)

    #-------------------------------------------------------------------------------------------------------
    # LRCN
    #-------------------------------------------------------------------------------------------------------

    # int     , Number of 2D CNN Layers
    p = [3,2,1]
    lrcn_num_cnn_layers.append((p[np.random.randint(len(p))]))
    
    # [int]   , per layer - Filer size at each layer
    p = [4,8,16,32,64,128]
    g = custom.draw_hp_contiguous_asc(p,3,lrcn_num_cnn_layers[-1])
    lrcn_cnn_filters_per_layer.append(g)

    # [(int)] , per layer - Strides per CNN layer
    p = [(2,2),(1,1)]
    g = custom.draw_hp_tuples_desc(p,lrcn_num_cnn_layers[-1])
    lrcn_cnn_strides_per_layer.append(g)

    # [(int)] , per layer - Kernal size per CNN layer
    p = [(3,3),(2,2)]
    g = custom.draw_hp_tuples_desc(p,lrcn_num_cnn_layers[-1])
    lrcn_cnn_kernal_size_per_layer.append(g)

    # [int]   , per-layer, do you want to batch normalize
    p = [1,0]
    g = list(np.repeat(p[np.random.randint(len(p))],lrcn_num_cnn_layers[-1]))
    lrcn_batchnormalize_per_layer.append(g)

    # int     , axis to perform batch normalization
    p = [-1]
    lrcn_batchnormalize_axis.append((p[np.random.randint(len(p))]))

    # float   , momentum for the moving mean and moviing variance
    p = [0.99]
    lrcn_batchnormalize_momentum.append((p[np.random.randint(len(p))]))

    # float   , small float aadded to variance to avoid dividing by 0
    p = [0.001]
    lrcn_batchnormalize_epsilon.append((p[np.random.randint(len(p))]))

    # Bool    , if True, add offset of beta to normalized tensor
    p = [True]
    lrcn_batchnormalize_center.append((p[np.random.randint(len(p))]))

    # Bool    , if True, multiply by gamma, if false, gamma is not used
    p = [True]
    lrcn_batchnormalize_scale.append((p[np.random.randint(len(p))]))

    # string  , initializer for the beta weights
    p = ['zeros']
    lrcn_batchnormalize_beta_init.append((p[np.random.randint(len(p))]))

    # string  , initializer for the gamma weigths
    p = ['ones']
    lrcn_batchnormalize_gamma_init.append((p[np.random.randint(len(p))]))

    # string  , initializer for the moving mean
    p = ['zeros']
    lrcn_batchnormalize_moving_mean.append((p[np.random.randint(len(p))]))

    # string  , initializer for the moving variance 
    p = ['ones']
    lrcn_batchnormalize_moving_var.append((p[np.random.randint(len(p))]))

    # [string], per layer - Activation per CNN layer
    p = ['relu']
    g = custom.draw_hp_with_missing(p,lrcn_num_cnn_layers[-1])
    lrcn_activation_type_per_layer.append(g)

    # [(int]) , per layer - Maxpooling poolsize per layer
    p = [(2,2),[]]
    g = custom.draw_hp_tuples_desc_with_missing(p,lrcn_num_cnn_layers[-1])
    lrcn_cnn_poolsize_per_layer.append(g)

    # [(int]) , per layer - Maxpooling poolstrides per layer
    p = [(2,2),(1,1)]
    g = custom.draw_hp_tuples_desc(p,lrcn_num_cnn_layers[-1])
    lrcn_cnn_poolstrides_per_layer.append(g)

    # [int]   , per layer - Dropouts 
    p = [.1,.2,.3,.4,.5,.6,0.0]
    g = custom.draw_hp_with_missing(p,lrcn_num_cnn_layers[-1])
    lrcn_dropouts_per_layer.append(g)

    # float   , Final dropout in the LSTM
    p = [.1,.2,.3,.4,.5,.6, 0.0]
    lrcn_pre_lstm_dropout.append((p[np.random.randint(len(p))]))

    # float
    p = [.1,.2,.3,.4,.5,.6,0.0]
    lrcn_lstm_dropout.append((p[np.random.randint(len(p))]))


    # float   , Fraction of recurrent connection to drop 
    p = [.1,.2,.3,.4,.5,.6,0.0]
    lrcn_lstm_recurrent_dropout.append((p[np.random.randint(len(p))]))

    # int     , LSTM output dimention
    p = [2,4,8,16,32]
    lrcn_lstm_output_dim.append((p[np.random.randint(len(p))]))

    # string  , LSTM activation
    p = ['tanh','sigmoid']
    lrcn_lstm_activation.append((p[np.random.randint(len(p))]))

    # string  , LSTM recurrent activation
    p = ['hard_sigmoid','sigmoid']
    lrcn_lstm_recurrent_activation.append((p[np.random.randint(len(p))]))

    # string  , LSTM initializer
    p = ['orthogonal']
    lrcn_lstm_recurrent_initializer.append((p[np.random.randint(len(p))]))

    # bool    , weather bias vector is used.
    p = [False]
    lrcn_lstm_use_bias.append((p[np.random.randint(len(p))]))

    # string  , intializer for the weights matrix
    p = ['glorot_uniform']
    lrcn_ltsm_kernal_initializer.append((p[np.random.randint(len(p))]))

    # string  , intitializer for the bias
    p = ['Zeros']
    lrcn_lstm_bias_intitializer.append((p[np.random.randint(len(p))]))

    #-------------------------------------------------------------------------------------------------------
    # 2D CNN 
    #-------------------------------------------------------------------------------------------------------

    # int     , Number of Convolational layers in the 3D CNN
    p = [4,3,2,1]
    cnn2d_num_cnn_layers.append((p[np.random.randint(len(p))]))

    # [int]   , per layer - The number of filters
    p = [4,8,16,32,64,128]
    g = custom.draw_hp_contiguous_asc(p,3,cnn2d_num_cnn_layers[-1])
    cnn2d_filters_per_layer.append(g)

    # [(int)] , per-layer - The kernal size of each filter
    p = [(4,4),(3,3),(2,2)]
    g = custom.draw_hp_tuples_desc(p,cnn2d_num_cnn_layers[-1])
    cnn2d_kernal_sizes_per_layer.append(g)

    # [(int)] , per-layer - The strides of each filter
    p = [(2,2),(1,1)]
    g = custom.draw_hp_tuples_desc(p,cnn2d_num_cnn_layers[-1])
    cnn2d_strides_per_layer.append(g)

    # [int]   , per-layer, do you want to batch normalize
    p = [1,0]
    g = list(np.repeat(p[np.random.randint(len(p))],cnn2d_num_cnn_layers[-1]))
    cnn2d_batchnormalize_per_layer.append(g)

    # int     , axis to perform batch normalization
    p = [-1]
    cnn2d_batchnormalize_axis.append((p[np.random.randint(len(p))]))

    # float   , momentum for the moving mean and moviing variance
    p = [0.99]
    cnn2d_batchnormalize_momentum.append((p[np.random.randint(len(p))]))

    # float   , small float aadded to variance to avoid dividing by 0
    p = [0.001]
    cnn2d_batchnormalize_epsilon.append((p[np.random.randint(len(p))]))

    # Bool    , if True, add offset of beta to normalized tensor
    p = [True]
    cnn2d_batchnormalize_center.append((p[np.random.randint(len(p))]))

    # Bool    , if True, multiply by gamma, if false, gamma is not used
    p = [True]
    cnn2d_batchnormalize_scale.append((p[np.random.randint(len(p))]))

    # string  , initializer for the beta weights
    p = ['zeros']
    cnn2d_batchnormalize_beta_init.append((p[np.random.randint(len(p))]))

    # string  , initializer for the gamma weigths
    p = ['ones']
    cnn2d_batchnormalize_gamma_init.append((p[np.random.randint(len(p))]))

    # string  , initializer for the moving mean
    p = ['zeros']
    cnn2d_batchnormalize_moving_mean.append((p[np.random.randint(len(p))]))

    # string  , initializer for the moving variance
    p = ['ones']
    cnn2d_batchnormalize_moving_var.append((p[np.random.randint(len(p))]))

    # [string], per-layer - The activation functions that are take the output of the CNN
    p = ['relu']
    g = custom.draw_hp_with_missing(p,cnn2d_num_cnn_layers[-1])
    cnn2d_activations_per_layer.append(g)

    # [(int)] , per-layer - The pool-size for the max-pooling later, after activiation.
    p = [(2,2),[]]
    g = custom.draw_hp_tuples_desc_with_missing(p,cnn2d_num_cnn_layers[-1])
    cnn2d_poolsize_per_layer.append(g)

    # [float] , per-layer - The dropouts, after maxpooling
    p = [.1,.2,.3,.4,.5,.6, []]
    g = custom.draw_hp_with_missing(p,cnn2d_num_cnn_layers[-1])
    cnn2d_dropouts_per_layer.append(g)

    # int     , The number of dense layers, after flattening the convolutionalal layers output
    p = [3,2,1]
    cnn2d_num_dense_layers.append((p[np.random.randint(len(p))]))

    # [int]   , per-layer - The number of nodes in the dense layer
    p = [4,8,16,32,64,128]
    g = custom.draw_hp_with_missing(p,cnn2d_num_dense_layers[-1])
    cnn2d_final_denselayer_size.append(g)

    # [string], per-layer - The activation of each dense layer
    p = ['relu']
    g = custom.draw_hp_with_missing(p,cnn2d_num_dense_layers[-1])
    cnn2d_final_denselayer_activation.append(g)

    # [int]   , per-layer - The dropout after activation of the desnse layer   
    p = [.1,.2,.3,.4,.5,.6,[]]
    g = custom.draw_hp_with_missing(p,cnn2d_num_dense_layers[-1])
    cnn2d_final_dropout.append(g)





# --- Experiment Number ---------------------------------------------------------------------------------------------------------------------------------------
hp = {'experiment'                      : experiment,                               # string, Name of result directory set to "str(time.time())", for multiple runs 
      'plot_perf'                       : plot_perf,                                # int   , Generate a plot of the performanc over time.  
      # --- Data Parameters -------------------------------------------------------------------------------------------------------------------------------------
    'take_only'                         : take_only,                                # float, % of the total data you want to keep
    'input_shape'                       : input_shape,                              # tuple, the shape of the input data
    'seed'                              : seed,                                     # int  , Random Seed        
    'train_perc'                        : train_perc,                               # flaot, Training percentage
    'val_perc'                          : val_perc,                                 # float, Validation percentage
    'test_perc'                         : test_perc,                                # float, Test percentage
    'checkpoint_period'                 : checkpoint_period,                        # int  , >= 1 , Epochs before we saving our weights
    'number_of_folds'                   : number_of_folds,                          # int  , >= 1 , Number of Cross validation folds
    'num_epochs'                        : number_of_epochs,                         # int  , >= 1 , Number of training data epochs 
    'batch_perc'                        : batch_perc,                               # float, Percentage of training to use data per training batch (1 = 100%) 
    'balance_class_cost'                : balance_class_cost,                       # bool , T|F  , weights data balance misclassificaiton costs.
    'class_weights'                     : class_weights,                            # [float], The weights of classes 0 and 1  
    # --- Weights initialization ----------------------------------------------# Web  ,  Initializers  : https://keras.io/initializers/-------------------------
    'weights_initialization'            : weights_initialization,                   # string, he_uniform|lecun_normal|he_normal|glorot_uniform|glorot_normal|lecun_uniform|Zeros|Ones|RandomNormal|RandomUniform|TruncatedNormal|VarianceScaling|Orthogonal
    'bias_initialization'               : bias_initialization,                      # string, he_uniform|lecun_normal|he_normal|glorot_uniform|glorot_normal|lecun_uniform|Zeros|Ones|RandomNormal|RandomUniform|TruncatedNormal|VarianceScaling|Orthogonal 
    # --- Loss Fucntion -------------------------------------------------------# Web  , https://keras.io/losses/ -----------------------------------------------
    'loss_function'                     : loss_function,                                         
    # --- For Optimization-----------------------------------------------------# Web  , https://keras.io/optimizers/ -------------------------------------------
    'optimizer'                         : optimizer,                                # string, sgd|rmsprop|adagrad|adadelta|adam|adamaz|nadam 
    'lr'                                : lr,                                       # float, >= 0, the learning rate of the optimizer 
    'clipnorm'                          : clipnorm,
    'clipvalue'                         : clipvalue,
    'decay'                             : decay,                                    # float, >= 0, the decay in the learning rate, per epoch
    'momentum'                          : momentum,                                 # float, >= 0, momentum parameter during sgd
    'rho'                               : rho,                                      # float, >= 0
    'epsilon'                           : epsilon,                                  # float, >= 0
    'beta_1'                            : beta_1,                                   # float, 0 < beta < 1
    'beta_2'                            : beta_2,                                   # float, 0 < beta < 1
    'nesterov'                          : nesterov,                                 # bool , T|F
    'amsgrad'                           : amsgrad,                                  # bool , T|F
    # --- For Callbacks -------------------------------------------------------# Web  , https://keras.io/callbacks/ --------------------------------------------                                  
    'early_stop_min_delta'              : early_stop_min_delta,                     # float, minimum amount things must change before we start applying the patience.
    'early_stop_patience'               : early_stop_patience,                      # int  ,  number of epochs with no improvement after which training will be stopped.
    'reduce_lr_patience'                : reduce_lr_patience,                       # float, How long must performance platteaus before lr is reduced  
    'reduce_lr_factor'                  : reduce_lr_factor,                         # flaot, Amount to reduce the loarning rate by: .2 is a factor of 5 
    'reduce_lr_epsilon'                 : reduce_lr_epsilon,                        # float, threshold for measuring new optimum
    'reduce_lr_cooldown'                : reduce_lr_cooldown,                       # int  , Number ofepochs to keep at this lr before resuming normal activity
    'reduce_lr_min_lr'                  : reduce_lr_min_lr,                         # float, minimum learning rate we can reduce to
    # --- For 3D CNN    ---------------------------------------------------------------------------------------------------------------------------------------
    # --- structure: CNN layers (CNN --> activation -> maxpooling --> dropouts) --> Flatten --> Dense Layers (Dense --> activation --> dropout) --> binary outcome
    # --- use [] to exclude a layer -------------------------------------------------------------------------------------------------------------------------- 
    'use_CNN_3D'                        : use_CNN_3D,                             # int      , Set to 1 if you want to use this model
    '3dcnn_num_layers'                  : cnn3d_num_layers,                       # int      . Number of Convolational layers in the 3D CNN
    '3dcnn_filters_per_layer'           : cnn3d_filters_per_layer,                # [int]    , per layer -Number of CNN filters
    '3dcnn_kernal_sizes_per_layer'      : cnn3d_kernal_sizes_per_layer,           # [(int)]  , per layer - CNN Kernal sizes
    '3dcnn_strides_per_layer'           : cnn3d_strides_per_layer,                # [(int)]  , per layer - CNN Strides
    '3dcnn_batchnormalize_per_layer'    : cnn3d_batchnormalize_per_layer,         # [int]   , per-layer, do you want to batch normalize
    '3dcnn_batchnormalize_axis'         : cnn3d_batchnormalize_axis,              # int     , axis to perform batch normalization
    '3dcnn_batchnormalize_momentum'     : cnn3d_batchnormalize_momentum,          # float   , momentum for the moving mean and moviing variance
    '3dcnn_batchnormalize_epsilon'      : cnn3d_batchnormalize_epsilon,           # float   , small float aadded to variance to avoid dividing by 0
    '3dcnn_batchnormalize_center'       : cnn3d_batchnormalize_center,            # Bool    , if True, add offset of beta to normalized tensor
    '3dcnn_batchnormalize_scale'        : cnn3d_batchnormalize_scale,             # Bool    , if True, multiply by gamma, if false, gamma is not used
    '3dcnn_batchnormalize_beta_init'    : cnn3d_batchnormalize_beta_init,         # string  , initializer for the beta weights
    '3dcnn_batchnormalize_gamma_init'   : cnn3d_batchnormalize_gamma_init,        # string  , initializer for the gamma weigths
    '3dcnn_batchnormalize_moving_mean'  : cnn3d_batchnormalize_moving_mean,       # string  , initializer for the moving mean
    '3dcnn_batchnormalize_moving_var'   : cnn3d_batchnormalize_moving_var,        # string  , initializer for the moving variance
    '3dcnn_activations_per_layer'       : cnn3d_activations_per_layer,            # [string] , per layer - CNN activation
    '3dcnn_poolsize_per_layer'          : cnn3d_poolsize_per_layer,               # [(int)]  , per layer - CNN maxpooling pool-size
    '3dcnn_dropouts_per_layer'          : cnn3d_dropouts_per_layer,               # [float]  , per layer -  CNN dropouts
    '3dcnn_num_dense_layers'            : cnn3d_num_dense_layers,                 # int      , Number of dense layers at the very end
    '3dcnn_final_denselayer_size'       : cnn3d_final_denselayer_size,            # [int]    , per layer - Number of nodes in the final dense layer
    '3dcnn_final_denselayer_activation' : cnn3d_final_denselayer_activation,      # [string] , per layer - Activation fucntion of the final layer
    '3dcnn_final_dropout'               : cnn3d_final_dropout,                    # [float]  , per layer - Dropout of the final layer
    
    # --- For LRCN    -----------------------------------------------------------------------------------------------------------------------------------------
    # --- structure: TIME Distributed CNN layers (CNN --> activation -> maxpooling --> dropouts) --> Flatten --> LSTM           
    # --- use [] to exclude a layer -----------------------------------------------------------------------------------------------------------------------------    
    'use_LRCN'                          : use_LRCN,                               # int     , Set to 1 if you want to use this model for this experiment         
    'lrcn_num_cnn_layers'               : lrcn_num_cnn_layers,                    # int     , Number of 2D CNN Layers
    'lrcn_cnn_filters_per_layer'        : lrcn_cnn_filters_per_layer,             # [int]   , per layer - Filer size at each layer
    'lrcn_cnn_strides_per_layer'        : lrcn_cnn_strides_per_layer,             # [(int)] , per layer - Strides per CNN layer
    'lrcn_cnn_kernal_size_per_layer'    : lrcn_cnn_kernal_size_per_layer,         # [(int)] , per layer - Kernal size per CNN layer
    'lrcn_batchnormalize_per_layer'     : lrcn_batchnormalize_per_layer,          # [int]   , per-layer, do you want to batch normalize
    'lrcn_batchnormalize_axis'          : lrcn_batchnormalize_axis,               # int     , axis to perform batch normalization
    'lrcn_batchnormalize_momentum'      : lrcn_batchnormalize_momentum,           # float   , momentum for the moving mean and moviing variance
    'lrcn_batchnormalize_epsilon'       : lrcn_batchnormalize_epsilon,            # float   , small float aadded to variance to avoid dividing by 0
    'lrcn_batchnormalize_center'        : lrcn_batchnormalize_center,             # Bool    , if True, add offset of beta to normalized tensor
    'lrcn_batchnormalize_scale'         : lrcn_batchnormalize_scale,              # Bool    , if True, multiply by gamma, if false, gamma is not used
    'lrcn_batchnormalize_beta_init'     : lrcn_batchnormalize_beta_init,          # string  , initializer for the beta weights
    'lrcn_batchnormalize_gamma_init'    : lrcn_batchnormalize_gamma_init,         # string  , initializer for the gamma weigths
    'lrcn_batchnormalize_moving_mean'   : lrcn_batchnormalize_moving_mean,        # string  , initializer for the moving mean
    'lrcn_batchnormalize_moving_var'    : lrcn_batchnormalize_moving_var,         # string  , initializer for the moving variance 
    'lrcn_activation_type_per_layer'    : lrcn_activation_type_per_layer,         # [string], per layer - Activatio nper CNN layer
    'lrcn_cnn_poolsize_per_layer'       : lrcn_cnn_poolsize_per_layer,            # [(int]) , per layer - Maxpooling poolsize per layer
    'lrcn_cnn_poolstrides_per_layer'    : lrcn_cnn_poolstrides_per_layer,         # [(int]) , per layer - Maxpooling poolstrides per layer
    'lrcn_dropouts_per_layer'           : lrcn_dropouts_per_layer,                # [int]   , per layer - Dropouts 
    'lrcn_pre_lstm_dropout'             : lrcn_pre_lstm_dropout,                  # float   , Final dropout in the LSTM
    'lrcn_lstm_dropout'                 : lrcn_lstm_dropout,                      #
    'lrcn_lstm_recurrent_dropout'       : lrcn_lstm_recurrent_dropout,            # float   , Fraction of recurrent connection to drop  
    'lrcn_lstm_output_dim'              : lrcn_lstm_output_dim,                   # int     , LSTM output dimention
    'lrcn_lstm_activation'              : lrcn_lstm_activation,                   # string  , LSTM activation
    'lrcn_lstm_recurrent_activation'    : lrcn_lstm_recurrent_activation,         # string  , LSTM recurrent activation
    'lrcn_lstm_recurrent_initializer'   : lrcn_lstm_recurrent_initializer,        # string  , LSTM initializer
    'lrcn_lstm_use_bias'                : lrcn_lstm_use_bias,                     # bool    , weather bias vector is used.
    'lrcn_ltsm_kernal_initializer'      : lrcn_ltsm_kernal_initializer,           # string  , intializer for the weights matrix
    'lrcn_lstm_bias_intitializer'       : lrcn_lstm_bias_intitializer,            # string  , intitializer for the bias

    # --- For 2D CNN  -----------------------------------------------------------------------------------------------------------------------------------------
    # --- structure: CNN layers (CNN --> activation -> maxpooling --> dropouts) --> Flatten --> Dense Layers (Dense --> activation --> dropout) --> binary outcome
    # --- use [] to exclude a layer ----------------------------------------------------------------------------------------------------------------------------
    'use_CNN_2D'                        : use_CNN_2D,                               # bool    , Set to 1 if you want to use this model, for this experiment
    'cnn2d_num_cnn_layers'              : cnn2d_num_cnn_layers,                     # int     , Number of Convolational layers in the 3D CNN
    'cnn2d_filters_per_layer'           : cnn2d_filters_per_layer,                  # [int]   , per layer - The number of filters
    'cnn2d_kernal_sizes_per_layer'      : cnn2d_kernal_sizes_per_layer,             # [(int)] , per-layer - The kernal size of each filter
    'cnn2d_strides_per_layer'           : cnn2d_strides_per_layer,                  # [(int)] , per-layer - The strides of each filter
    'cnn2d_batchnormalize_per_layer'    : cnn2d_batchnormalize_per_layer,           # [int]   , per-layer, do you want to batch normalize
    'cnn2d_batchnormalize_axis'         : cnn2d_batchnormalize_axis,                # int     , axis to perform batch normalization
    'cnn2d_batchnormalize_momentum'     : cnn2d_batchnormalize_momentum,            # float   , momentum for the moving mean and moviing variance
    'cnn2d_batchnormalize_epsilon'      : cnn2d_batchnormalize_epsilon,             # float   , small float aadded to variance to avoid dividing by 0
    'cnn2d_batchnormalize_center'       : cnn2d_batchnormalize_center,              # Bool    , if True, add offset of beta to normalized tensor
    'cnn2d_batchnormalize_scale'        : cnn2d_batchnormalize_scale,               # Bool    , if True, multiply by gamma, if false, gamma is not used
    'cnn2d_batchnormalize_beta_init'    : cnn2d_batchnormalize_beta_init,           # string  , initializer for the beta weights
    'cnn2d_batchnormalize_gamma_init'   : cnn2d_batchnormalize_gamma_init,          # string  , initializer for the gamma weigths
    'cnn2d_batchnormalize_moving_mean'  : cnn2d_batchnormalize_moving_mean,         # string  , initializer for the moving mean
    'cnn2d_batchnormalize_moving_var'   : cnn2d_batchnormalize_moving_var,          # string  , initializer for the moving variance
    'cnn2d_activations_per_layer'       : cnn2d_activations_per_layer,              # [string], per-layer - The activation functions that are take the output of the CNN
    'cnn2d_poolsize_per_layer'          : cnn2d_poolsize_per_layer,                 # [(int)] , per-layer - The pool-size for the max-pooling later, after activiation.
    'cnn2d_dropouts_per_layer'          : cnn2d_dropouts_per_layer,                 # [float] , per-layer - The dropouts, after maxpooling
    'cnn2d_num_dense_layers'            : cnn2d_num_dense_layers,                   # int     , The number of dense layers, after flattening the convolutionalal layers output
    'cnn2d_final_denselayer_size'       : cnn2d_final_denselayer_size,              # [int]   , per-layer - The number of nodes in the dense layer
    'cnn2d_final_denselayer_activation' : cnn2d_final_denselayer_activation,        # [string], per-layer - The activation of each dense layer
    'cnn2d_final_dropout'               : cnn2d_final_dropout,                      # [int]   , per-layer - The dropout after activation of the desnse layer   
    }

np.save('hyper_parameters.npy',hp)


