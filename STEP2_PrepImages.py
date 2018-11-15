#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:36:44 2018

@author: mohammad
"""
#------------------------------------------------------------------------------
# Import the relevant libraries
#------------------------------------------------------------------------------
import scipy.io
import numpy as np
import random

#------------------------------------------------------------------------------
# Importing the data and the mask  
#------------------------------------------------------------------------------
ids = np.load('python_covariates/covars_ids.npy')
covars = np.load('python_covariates/covars.npy')
mask = np.load('python_covariates/mask.npy') 

#------------------------------------------------------------------------------
# Fix the names of the files.
#------------------------------------------------------------------------------
newids=[]
for i in range(np.size(ids,0)):
    name = ids[i][0]
    name = name[7:-4]
    newids.append('python_images/' + name + '.npy')
python_file_locs = np.array(newids)    
np.save('python_covariates/python_file_locs',python_file_locs)

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
freq_lim = 50
time_cut_off_mins = 72*60
downsample_by = 2
numsamples = 600

for i in range(np.size(ids,0)):
    
    #load the data
    mat = scipy.io.loadmat(ids[i][0]) 

    #get the name for saving!
    name = ids[i][0]
    name = name[7:-4]
    
    # Extract the data                                     
    dat = mat['F']  

    #cast any nans as zeros
    dat[np.isnan(dat)] = 0

    # Convert data to float32, and normalize to be between 0 and 1.
    dat = dat.astype('float32')
    dat /=255
    
    
    # Apply the mask to the entire dataset
    masknd = np.repeat(mask[:, :, np.newaxis],freq_lim, axis=2)
    masknd = np.repeat(masknd[:, :, :,np.newaxis],np.size(dat,3), axis=3) 
    dat = masknd*dat
    
    delta = np.sum(dat[:,:,0:3,:],2)/4
    theta = np.sum(dat[:,:,4:7,:],2)/4
    alpha = np.sum(dat[:,:,8:15,:],2)/8
    beta  = np.sum(dat[:,:,16:31,:],2)/16

    # Extract the frequency bands
    dat_tmp = np.zeros([np.size(dat,0),np.size(dat,1),4,np.size(dat,3)])
    dat_tmp[:,:,0,:] = delta 
    dat_tmp[:,:,1,:] = theta 
    dat_tmp[:,:,2,:] = alpha 
    dat_tmp[:,:,3,:] = beta     
    dat              = dat_tmp 

    #downsample
    dat = dat[xrange(0,np.size(dat,0),3),:,:,:]
    dat = dat[:, xrange(0,np.size(dat,1),3),:,:]

    #throw out the unnecesary bits on the tail ends.
    dat = dat[1:9,1:9,:,:]

    # Extract the offset (time between arrest and eeg start time)
    this_offset = covars[i,3]
    
    # pad the data with zeros so that it is of a consistent length.
    padding_size = int(60*this_offset)
    front_padding = np.zeros([8,8,4,padding_size])

    padding_size = time_cut_off_mins - (int(60*this_offset) + np.size(dat,3))
    if padding_size > 0:
        end_padding = np.zeros([8,8,4,padding_size])  
        dat = np.append(front_padding,dat,axis=3)
        dat = np.append(dat,end_padding,axis=3)
    else:
        dat = np.append(front_padding,dat,axis=3)
        dat = dat[:,:,:,0:time_cut_off_mins]

    # transform data to keras firendly format
    dat = dat.transpose(3,0,1,2)
    dupdat = np.zeros([numsamples,72,8,8,4])
    
    #how many hours of data do we have?
    for n in range(0,numsamples):  
        base = 0
        t_index = []
        for t in range(0,72): 
            t_index.append(random.randint(0,59)+base)
            base = base + 60
           
        dupdat[n,:,:,:,:] = dat[np.array(t_index),:,:,:]


    dat = dupdat

#     # Visualize the images.
#    fig,axarr = plt.subplots(2, 2, dpi=100)
#    axarr[0,0].imshow(dat[20,9,:,:,0])
#    axarr[0,0].set_title('delta power', fontsize=8)
#    axarr[0,1].imshow(dat[20,9,:,:,1])
#    axarr[0,1].set_title('theta power', fontsize=8)
#    axarr[1,0].imshow(dat[20,9,:,:,2])
#    axarr[1,0].set_title('alpha power', fontsize=8)
#    axarr[1,1].imshow(dat[20,9,:,:,3])
#    axarr[1,1].set_title('beta power', fontsize=8)

###############################################################################
## TRAINING DATA ##############################################################
# Subjects x Time x rows x columns x Hertz
###############################################################################
    np.save('python_images/' + name,dat)
