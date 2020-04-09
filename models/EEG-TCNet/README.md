Copyright (C) 2020 ETH Zurich, Switzerland. SPDX-License-Identifier: CC-BY-ND-4.0. See LICENSE file for details.

Authors: Thorir Mar Ingolfsson, Michael Hersche, Xiaying Wang, Nobuaki Kobayashi, Lukas Cavigelli, Luca Benini

# EEG-TCNet models

### Definiton of structure
Here are 9 subdirectories for 9 different subjects, `S1` which stands for Subject 1, within each subdirectory are four .h5 files.\
`model.h5` is the pretrained Keras model of the Variable EEG-TCNet\
`model_fixed.h5` is the pretrained Keras model of the EEG-TCNet\
`pipeline.h5` is a pipeline that includes data normalization (if applied) before using the pretrained model of Variable EEG-TCNet\
`pipeline_fixed.h5` is a pipeline that includes data normalization (if applied) before using the pretrained model of EEG-TCNet

### Usage
Please refer to `Accuracy_and_kappa_scores.ipynb` to see how the models are loaded and validated.

### License and Attribution
Everything in these subdirectories is licensed under a Creative Commons Attribution-NoDerivatives 4.0 International License. Please refer to the LICENSE file.