#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:00:19 2018

@author: mohammad
"""
import sklearn
import numpy as np
import pylab
from sklearn.model_selection import train_test_split
import matplotlib as plt
import os
import json

def gen_folders(time_dir,number_of_folds):                            
    os.mkdir('Checkpoints/' + time_dir)                       # make the directory '<time_val>'
    os.mkdir('Checkpoints/' + time_dir + '/topology')         # make the directory 'topology'
    os.mkdir('Checkpoints/' + time_dir +'/all_performance')          # make the directory 'overall'
    
    for fold_num in range(0,number_of_folds): 
        os.mkdir('Checkpoints/' + time_dir + '/fold' + str(fold_num))                      # make the director for this fold
        os.mkdir('Checkpoints/' + time_dir + '/fold' + str(fold_num) + '/weights')         # make the directory for weights
        os.mkdir('Checkpoints/' + time_dir + '/fold' + str(fold_num) + '/performance')     # make the directory for performance


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
    n_samples = np.size(eg_data,0)
    n_frames = np.size(eg_data,1)
    row = np.size(eg_data,2)
    col = np.size(eg_data,3)
    n_hertz = np.size(eg_data,4)

    #Initialize the data you want to return
    Y_test = np.zeros(int(n_samples*np.size(test_ind)*take_only))
    X_test = np.zeros([int(n_samples*np.size(test_ind)*take_only),
                       n_frames,
                       row,
                       col,
                       n_hertz])
    
    # append the data together.
    for i in range(0,np.size(testing_files)):
        this_data = np.load(testing_files[i].decode('utf8'))
        X_test[ind:int(ind+n_samples*take_only),:,:,:,:] = this_data[0:int(np.size(this_data,0)*take_only),:,:,:,:]
        Y_test[ind:int(ind+n_samples*take_only)] = outcomes[i]*np.ones(int(np.size(this_data,0)*take_only))
        ind = int(ind+n_samples*take_only) 
    
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


# EVALUATES THE MODEL ACCORDING TO SEVERAL PERFORMANCE METRICS.
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


