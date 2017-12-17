"""
==========================================================
Written by Dominic Masters, Graphcore, dominic.masters@graphcore.ai
==========================================================
"""
import tensorflow as tf
import sys
import numpy as np
import random
import TF_IM_CLASS.Networks.Tools as N

class SyntheticDataset(object):
  # class for use with static datasets

  def __init__(self, data_shape, num_classes, valid_size=1000, one_hot=True, dtype=tf.float32, inttype=tf.int32, seed=None):
    dim=len(data_shape)
    self.dim=dim
    self.data_shape=data_shape
    self.num_classes=num_classes
    self.num_train_ex=1
    self.batch_size=self.data_shape[0]
    self.valid_size=valid_size
    self.one_hot=one_hot
    self.dtype=dtype
    self.inttype=inttype

    self.feed_dict=True

    if seed==None:
      seed=np.random.randint(low=0, high=2**32)
    self.seed=seed
    np.random.seed(self.seed); random.seed(self.seed); tf.set_random_seed(self.seed)
    print('Data Seed: ' + str(seed))

    if sys.version_info < (3, 0):
      print('\nDataset parameters:')
      for k, v in list(locals().iteritems()):
        if not k=='DATA':
          print(k + ' = ' + str(v))

  def initialise_tf_vars(self, calc_train_acc=True, calc_valid_acc=True, calc_test_acc=True):

    self.tf_batch_dataset = tf.constant(np.random.normal(scale=1e-1, size=self.data_shape), dtype=self.dtype)
    labels=np.random.randint(0, self.num_classes, self.data_shape[0])
    if self.one_hot:
      labels=np.eye(self.num_classes)[labels]
    self.tf_batch_labels = tf.constant(labels, dtype=self.inttype)

    if calc_valid_acc==True:
      valid_shape=self.data_shape
      valid_shape[0]=self.valid_size
      self.tf_valid_dataset = tf.constant(np.random.normal(scale=1e-1, size=valid_shape), dtype=self.dtype)
      labels=np.random.randint(0, self.num_classes, valid_shape[0])
      if self.one_hot:
        labels=np.eye(self.num_classes)[labels]
      self.tf_valid_labels = tf.constant(labels, dtype=self.inttype)

  def get_next_batch(self):
    return {}
