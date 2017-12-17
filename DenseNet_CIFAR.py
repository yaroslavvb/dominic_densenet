"""
==========================================================
Written by Dominic Masters, Graphcore, dominic.masters@graphcore.ai
==========================================================
"""

import Session as IC
import tensorflow as tf
import numpy as np
import random as random
#from FakeDataset import FakeDataset
import DenseNet
from Static_Dataset import Static_Dataset
import Augmentation as Aug
import CIFAR10_data as CIFAR10

DATA = CIFAR10.create_datasets(train_size = 50000, valid_size = 0, test_size = 10000,
  normalise=True, compact=True, test_as_valid=True,
  normalisation_method='per_pixel', shape='4D')

inference=lambda input_data, labels, **nargs : DenseNet.inference(input_data, labels,
 preset_config='CIFAR_100', **nargs)

tot_it=120000

D=Static_Dataset(DATA, batch_size=64)

learning_r = lambda global_step: tf.train.piecewise_constant(global_step,
  [tot_it/2, 3*tot_it/4], [ 0.05, 0.005, 0.0005])

augmentation_fun = lambda images: Aug.pad_flip_and_random_crop(images,padding=4)

optimiser={'fun':tf.train.MomentumOptimizer, 'args':[0.9], 'narg':{'use_nesterov': True}}

S=IC.TF_session(D,inference, learning_r, optimiser=optimiser, augmentation_fun=augmentation_fun,
  log_dir='./logs/CIFAR_DenseNet/', name='DenseNet100_full', calc_valid_acc=True, calc_train_acc=True)

S.run(tot_it,2)

