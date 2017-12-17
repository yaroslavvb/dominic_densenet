"""
==========================================================
Written by Dominic Masters, Graphcore, dominic.masters@graphcore.ai
==========================================================
"""

import Session as IC
import tensorflow as tf
import numpy as np
from SyntheticDataset import SyntheticDataset
import random as random
import DenseNet
from Static_Dataset import Static_Dataset
import Augmentation as Aug
import CIFAR10_data as CIFAR10

inference=lambda input_data, labels, **nargs : DenseNet.inference(input_data, labels,
 preset_config='ImageNet_121', **nargs)

tot_it=1000

D = SyntheticDataset([8,224,224,3],1000)

augmentation_fun = lambda images: Aug.pad_flip_and_random_crop(images,padding=4)

optimiser={'fun':tf.train.MomentumOptimizer, 'args':[0.9], 'narg':{'use_nesterov': True}}

S=IC.TF_session(D,inference, 0.01, optimiser=optimiser, augmentation_fun=augmentation_fun,
  log_dir='./logs/ImageNetSize_DenseNet/', name='DenseNet121', calc_valid_acc=True, calc_train_acc=True)

S.run(tot_it,100)

