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
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*

#!/usr/bin/env python3

from utils.models import EEGTCNet
from utils.data_loading import prepare_features
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import numpy as np

data_path = 'data/'
F1 = 8
KE = 32
KT = 4
L = 2
FT = 12
pe = 0.2
pt = 0.3
classes = 4
channels = 22
crossValidation = False
batch_size = 64
epochs = 750
lr = 0.001
for subject in range(9):
    path = data_path+'s{:}/'.format(subject+1)
    X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot = prepare_features(path,subject,crossValidation)
    model = EEGTCNet(nb_classes = 4,Chans=22, Samples=1125, layers=L, kernel_s=KT,filt=FT, dropout=pt, activation='elu', F1=F1, D=2, kernLength=KE, dropout_eeg=pe)
    opt = Adam(lr=lr)
    model.compile(loss=categorical_crossentropy,optimizer=opt, metrics=['accuracy'])
    for j in range(22):
        scaler = StandardScaler()
        scaler.fit(X_train[:,0,j,:])
        X_train[:,0,j,:] = scaler.transform(X_train[:,0,j,:])
        X_test[:,0,j,:] = scaler.transform(X_test[:,0,j,:])
    model.fit(X_train, y_train_onehot,batch_size=batch_size, epochs=750, verbose=0)
    y_pred = model.predict(X_test).argmax(axis=-1)
    labels = y_test_onehot.argmax(axis=-1)
    accuracy_of_test = accuracy_score(labels, y_pred)
    print(accuracy_of_test)