Copyright (C) 2020 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file for details.

Authors: Thorir Mar Ingolfsson, Michael Hersche, Xiaying Wang, Nobuaki Kobayashi, Lukas Cavigelli, Luca Benini

# EEG-TCNet
This project provides the experimental environment used to produce the results reported in the paper *EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery Brain-Machine Interfaces* available on [arXiv](https://arxiv.org/abs/2006.00622). If you find this work useful in your research, please cite
```
@misc{ingolfsson2020eegtcnet,
      title={EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery Brain-Machine Interfaces}, 
      author={Thorir Mar Ingolfsson and Michael Hersche and Xiaying Wang and Nobuaki Kobayashi and Lukas Cavigelli and Luca Benini},
      year={2020},
      eprint={2006.00622},
      archivePrefix={arXiv},
      primaryClass={eess.SP}
}

```

## Getting started

### Prerequisites
* We developed and used the code behind EEG-TCNet on [Ubuntu 18.04.3 LTS (Bionic Beaver) (64bit)](http://old-releases.ubuntu.com/releases/18.04.3/).
* The code behind EEG-TCNet is based on Python3, and [Anaconda3](https://www.anaconda.com/distribution/) is required.
* We used [NVidia GTX1080 Ti GPUs](https://developer.nvidia.com/cuda-gpus) to accelerate the training of our models (driver version [396.44](https://www.nvidia.com/Download/driverResults.aspx/136950/en-us)). In this case, CUDA and the cuDNN library are needed (we used [CUDA 10.1](https://developer.nvidia.com/cuda-toolkit-archive)).

Also the dataset 2a of the BCI Competition IV needs to be downloaded and put into the `/data` folder. It is available on [here](http://bnci-horizon-2020.eu/database/data-sets)
### Installing
Navigate to EEG-TCNet's main folder and create the environment using Anaconda: (We have two environments (one for GPU and one for CPU)) for GPU do:
```
$ conda env create -f EEG-TCNet-GPU.yml -n EEG-TCNet
```
For CPU do
```
$ conda env create -f EEG-TCNet.yml -n EEG-TCNet
```


## Usage
We provide the models under `/models/EEG-TCNet` inside there we have 9 subdirectories `/S1` to `/S9` each representing each subject. Inside each subdirectory there are 6 files. `model.h5` is the saved keras model of variable EEG-TCNet and `model_fixed.h5` is the saved keras model of fixed EEG-TCNet. Then there are two pipeline files in each subdirectory which vary depending on if data normalization was used or not. Please refer to  `Accuracy_and_kappa_scores.ipynb` in the main directory to see how these pipelines are produces. There you also find the accuracy score and kappa score verification of EEG-TCNet.

Under `/utils` you find the data loading and model making files. Then also a small sample of how to train is given with `sample_train.py`, please note that because of the stochastic nature of training with GPUs it's very hard to fix every random variable in the backend. Therefore to reproduce the same or similar models one might need to train a couple of times in order to get the same highly accurate models we present.

### License and Attribution
Please refer to the LICENSE file for the licensing of our code.