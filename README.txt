-----------------------------------------
----  PRELIMINARIES   ----
-----------------------------------------
*** must make sure that cuda libraries are on the PATH ***
*** make sure that requirements.txt has tensorflow-gpu==1.4.0 ***

FOR PYTHON 3.6.0: virtualenv -p python3 venv
module load python/3.6.0
module load cuda/8.0
module load cudnn/7
source venv/bin/activate
(venv) pip install -r requirements.txt
(venv) STEP3_TrainNeuralNetwork.py
LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64:
PATH=/home/mohammad/anaconda2/bin:/usr/local/cuda-6.5/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:

deactivate

-----------------------------------------
----  STEP1_ImportMatCovariates.py   ----
-----------------------------------------
Imports the various covariates (age, cause, rosc, cpc, etc.), and saves them
as .npy files in the 'python_covariates' directory

-----------------------------------------
----    STEP2_PrepImages.py     ----
-----------------------------------------
Converts spatio-temporal topoplots in '/images' from matlab format to .npy 
files. Also reshapes the data so that it can be used by the Keras DNN framework
and samples each subject multiple times. 

-----------------------------------------
----  STEP3_TrainNeuralNet.py ----
-----------------------------------------
Train a 3D CNN
