#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:36:44 2018

@author: Mohammad Ghassemi
"""

#Import the relevant libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import scipy.io
import h5py
from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import math
import pylab

###############################################################################
## IMPORTING DATA #############################################################
###############################################################################

# Import the CPC values
imported = scipy.io.loadmat('matlab_covariates/cpcbest.mat');
cpcbest = imported['cpcbest'].astype('float32')[:,0]

imported = scipy.io.loadmat('matlab_covariates/cpc0.mat');
cpc0 = imported['cpc0'].astype('float32')[:,0]

imported = scipy.io.loadmat('matlab_covariates/cpc3.mat');
cpc3 = imported['cpc3'].astype('float32')[:,0]

imported = scipy.io.loadmat('matlab_covariates/cpc6.mat');
cpc6 = imported['cpc6'].astype('float32')[:,0]

imported = scipy.io.loadmat('matlab_covariates/vfib.mat');
vfib = imported['vfib'].astype('float32')[:,0]

imported = scipy.io.loadmat('matlab_covariates/sex.mat');
sex = imported['sex'].astype('float32')[:,0]

imported = scipy.io.loadmat('matlab_covariates/rosc.mat');
rosc = imported['rosc'].astype('float32')[:,0]

imported = scipy.io.loadmat('matlab_covariates/offset.mat');
offset = imported['offset'].astype('float32')[:,0]

imported = scipy.io.loadmat('matlab_covariates/location.mat');
location = imported['location'].astype('float32')[:,0]

imported = scipy.io.loadmat('matlab_covariates/cause.mat');
cause = imported['cause'].astype('float32')[:,0]

imported = scipy.io.loadmat('matlab_covariates/age.mat');
age = imported['age'].astype('float32')[:,0]
    
# Import the file locations
imported = scipy.io.loadmat('matlab_covariates/filelocation.mat');
imported = imported['filelocation']

filelist = []
for i in range(790):
    a = imported[i][0][0];
    filelist.append(a.encode('ascii','ignore'))
filelist = np.array(filelist)

#fi
imported = scipy.io.loadmat('matlab_covariates/eegid.mat');
imported = imported['eegid']
eegid = []
for i in range(790):
    a = imported[i][0][0];
    eegid.append(a.encode('ascii','ignore'))
eegid = np.array(eegid)
    
imported = scipy.io.loadmat('matlab_covariates/sid.mat');
imported = imported['sid']
sid = []
for i in range(790):
    a = imported[i][0][0];
    sid.append(a.encode('ascii','ignore'))    
sid = np.array(sid)
   

imported = scipy.io.loadmat('matlab_covariates/inst.mat');
imported = imported['inst']
inst = []
for i in range(790):
    a = imported[i][0][0];
    inst.append(a.encode('ascii','ignore'))    
inst = np.array(inst)
 
### FIND PEOPLE WITH MISSING OUTCOMES
toss = pylab.find(np.isnan(cpcbest))
np.size(toss)
age = np.delete(age,toss,0)
cause = np.delete(cause,toss,0)
cpc0 = np.delete(cpc0,toss,0)
cpc3 = np.delete(cpc3,toss,0)
cpc6 = np.delete(cpc6,toss,0)
cpcbest = np.delete(cpcbest,toss,0)
eegid = np.delete(eegid,toss,0)
filelist = np.delete(filelist,toss,0);
location = np.delete(location,toss,0);
offset = np.delete(offset,toss,0);
rosc = np.delete(rosc,toss,0);
sex = np.delete(sex,toss,0);
sid = np.delete(sid,toss,0);
vfib = np.delete(vfib,toss,0);
inst = np.delete(inst,toss,0);

### FIND PEOPLE WITH MISSING OFFSETS
toss = pylab.find(np.isnan(offset))
np.size(toss)

age = np.delete(age,toss,0)
cause = np.delete(cause,toss,0)
cpc0 = np.delete(cpc0,toss,0)
cpc3 = np.delete(cpc3,toss,0)
cpc6 = np.delete(cpc6,toss,0)
cpcbest = np.delete(cpcbest,toss,0)
eegid = np.delete(eegid,toss,0)
filelist = np.delete(filelist,toss,0);
location = np.delete(location,toss,0);
offset = np.delete(offset,toss,0);
rosc = np.delete(rosc,toss,0);
sex = np.delete(sex,toss,0);
sid = np.delete(sid,toss,0);
vfib = np.delete(vfib,toss,0);
inst = np.delete(inst,toss,0);

### FIND PEOPLE WITH CORUPTED OFFSETS
toss = pylab.find(offset > 71)
np.size(toss)

age = np.delete(age,toss,0)
cause = np.delete(cause,toss,0)
cpc0 = np.delete(cpc0,toss,0)
cpc3 = np.delete(cpc3,toss,0)
cpc6 = np.delete(cpc6,toss,0)
cpcbest = np.delete(cpcbest,toss,0)
eegid = np.delete(eegid,toss,0)
filelist = np.delete(filelist,toss,0);
location = np.delete(location,toss,0);
offset = np.delete(offset,toss,0);
rosc = np.delete(rosc,toss,0);
sex = np.delete(sex,toss,0);
sid = np.delete(sid,toss,0);
vfib = np.delete(vfib,toss,0);
inst = np.delete(inst,toss,0);

##COMPUTE THE BINARY OUTCOME
cpcbest_badoutcome = (cpcbest > 3).astype('float32');


#Merge all the data and save it for later.
covars  = np.column_stack((age,cause,location,offset,rosc,sex,vfib,
                     cpc0,cpc3,cpc6,cpcbest,cpcbest_badoutcome))

covar_headers = np.array(['age','cause','location','offset','rosc','sex','vfib',
                              'cpc0','cpc3','cpc6','cpcbest','cpcbest_badoutcome'])

covar_ids  =  np.column_stack((filelist,sid,eegid))

np.save('python_covariates/inst',inst)    
np.save('python_covariates/covars',covars)
np.save('python_covariates/covars_headers',covar_headers)
np.save('python_covariates/covars_ids',covar_ids)