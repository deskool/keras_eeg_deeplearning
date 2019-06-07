#!/usr/bin/env python3

"""
Created on Wed Feb 21 10:36:01 2018

@author: Tuka Alhanai (tuka@mit.edu)

Evaluating results of NN training
"""


import numpy as np
import time
import sys

Nexp     = sys.argv[1] # 300
fold_num = 5
outdir   = 'Checkpoints/' # 'Checkpoints/exp3/'

# load hyperparameters
# =============================================================================
print('==== Loading Hyperparameters')
hp = np.load("hyper_parameters.npy")
# hp = np.load('Step08_hyperparameters.npy')
hp = hp.item()

# load results
# =============================================================================
print('==== Loading performance results')
while True:
    auc = []
    f1 = []
    badh = ''
    for h in range(Nexp):
        
        if fold_num == 0:
            try:
                headers   = np.load(outdir + hp['experiment'][h]  + '/fold' + str(fold_num) + "/performance/metric_names.npy")
                test_perf = np.load(outdir + hp['experiment'][h]  + '/fold' + str(fold_num) + "/performance/test_performance.npy")
                auc.append(test_perf[-1,5])
                f1.append(test_perf[-1,16])
                
            except FileNotFoundError:
                badh = badh + str(h) + ',' 
                auc.append(0)
                f1.append(0)
        else:
              try:
                headers   = np.load(outdir + hp['experiment'][h]  + "/all_performance/metric_names.npy")
                test_perf = np.load(outdir + hp['experiment'][h]  + "/all_performance/test_performance.npy")
                auc.append(test_perf[-1,5])
                f1.append(test_perf[-1,16])
                
              except FileNotFoundError:
                badh = badh + str(h) + ',' 
                auc.append(0)
                f1.append(0)

            
    print('Tasks that are empty: ', badh)
    print('')
    
    
    # print key hyperparameters
    # =============================================================================
    print('==== Key Hyperparameters')
    sortInd = np.argsort(auc)[::-1]
    thresh = 0.5
    for h in sortInd:
        
        if auc[h] > thresh:
            print(str(h),
                  '[AUC]: ',        str(auc[h])[:5],
                  '[F1]: ',         str(f1[h])[:5],
                  '[Nlayers]: ',    hp['feedforward_num_layers'][h], 
                  '[Act]: ',        str(hp['feedforward_activation_per_layer'][h][0]).ljust(5),
                  '[D-Out]: ',      str(hp['feedforward_dropout_rate'][h][0]).ljust(4),
                  # '[B-Norm]: ',     str(hp['feedforward_batchnormalize_per_layer'][h][0]).ljust(2),
                  '[B-Size]: ',     str(hp['batch_size'][h]).ljust(7),
                  '[WInit]: ',      str(hp['weights_initialization'][h]).ljust(15),
                  # '[Opt]: ',        str(hp['optimizer'][h]).ljust(7),
                  '[LR]: ',         str(hp['lr'][h]).ljust(7),
                  '[Decay]: ',      str(hp['decay'][h][0]).ljust(7),
                  '[Mo]: ',         str(hp['momentum'][h]).ljust(5),
                  '[Loss]: ',       str(hp['loss_function'][h]).ljust(15),
                  '[Hsize]: ',    str(hp['feedforward_denselayer_size'][h]).ljust(15))
    
    print('\n')
    # time.sleep(10)

