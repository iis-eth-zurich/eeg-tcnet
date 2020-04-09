#*----------------------------------------------------------------------------*
#* Copyright (C) 2020 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Authors:  Thorir Mar Ingolfsson, Michael Hersche, Tino Rellstab            *
#*----------------------------------------------------------------------------*

#!/usr/bin/env python3

'''	Loads the dataset 2a of the BCI Competition IV
available on http://bnci-horizon-2020.eu/database/data-sets
'''
import numpy as np
import scipy.io as sio
import glob as glob
from tensorflow.keras.utils import to_categorical


def load_all_data (crossValidation, data_path): 

    big_X_train, big_y_train, big_X_test, big_y_test = [None]*9, [None]*9, [None]*9, [None]*9
    for subject in range (0,9):
        path = data_path+'s' + str(subject+1) + '/'
        big_X_train[subject], big_y_train[subject] = get_data(subject+1, True ,path)
        big_X_test[subject], big_y_test[subject] = get_data(subject+1, False ,path)
    
    return big_X_train, big_y_train, big_X_test, big_y_test

def get_data(subject,training,path, highpass = False):
	'''	Loads the dataset 2a of the BCI Competition IV
	available on http://bnci-horizon-2020.eu/database/data-sets

	Keyword arguments:
	subject -- number of subject in [1, .. ,9]
	training -- if True, load training data
				if False, load testing data
	
	Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
			class_return 	numpy matrix 	size = NO_valid_trial
	'''
	NO_channels = 22
	NO_tests = 6*48 	
	Window_Length = 7*250 

	class_return = np.zeros(NO_tests)
	data_return = np.zeros((NO_tests,NO_channels,Window_Length))

	NO_valid_trial = 0
	if training:
		a = sio.loadmat(path+'A0'+str(subject)+'T.mat')
	else:
		a = sio.loadmat(path+'A0'+str(subject)+'E.mat')
	a_data = a['data']
	for ii in range(0,a_data.size):
		a_data1 = a_data[0,ii]
		a_data2= [a_data1[0,0]]
		a_data3= a_data2[0]
		a_X 		= a_data3[0]
		a_trial 	= a_data3[1]
		a_y 		= a_data3[2]
		a_fs 		= a_data3[3]
		a_classes 	= a_data3[4]
		a_artifacts = a_data3[5]
		a_gender 	= a_data3[6]
		a_age 		= a_data3[7]

		for trial in range(0,a_trial.size):
			if(a_artifacts[trial]==0):
				data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
				class_return[NO_valid_trial] = int(a_y[trial])
				NO_valid_trial +=1


	return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]

def prepare_features(path,subject,crossValidation=False):

    fs = 250 
    t1 = int(1.5*fs)
    t2 = int(6*fs)
    T = t2-t1
    X_train, y_train = get_data(subject+1,True,path)
    if crossValidation:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=0)
    else:
        X_test, y_test = get_data(subject+1,False , path)

    # prepare training data 	
    N_tr,N_ch,_ =X_train.shape 
    X_train = X_train[:,:,t1:t2].reshape(N_tr,1,N_ch,T)
    y_train_onehot = (y_train-1).astype(int)
    y_train_onehot = to_categorical(y_train_onehot)
    # prepare testing data 
    N_test,N_ch,_ =X_test.shape 
    X_test = X_test[:,:,t1:t2].reshape(N_test,1,N_ch,T)
    y_test_onehot = (y_test-1).astype(int)
    y_test_onehot = to_categorical(y_test_onehot)	

    return X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot