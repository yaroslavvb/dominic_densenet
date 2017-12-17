"""
==========================================================
Written by Dominic Masters, Graphcore, dominic.masters@graphcore.ai
==========================================================
"""
import tensorflow as tf
import sys
import numpy as np
import random
#import TF_IM_CLASS.Networks.Tools as N

class Static_Dataset(object):
  # class for use with static datasets

  def __init__(self, DATA, batch_size, dtype=tf.float32, inttype=tf.int32, seed=None, convert_from_one_hot=False):
    self.batch_size=batch_size

    #unpack data
    self.DATA=DATA

    self.ind=[] # randomised training image indices left in epoch

    dim=dim=len(self.DATA['train_dataset'].shape)
    self.dim=dim
    self.data_shape=list(self.DATA['train_dataset'].shape)
    self.data_shape[0]=batch_size
    self.num_classes=self.DATA['train_labels'].shape[1]
    self.num_train_ex=self.DATA['train_dataset'].shape[0]

    self.dtype=dtype
    self.inttype=inttype

    self.feed_dict=True
    self.convert_from_one_hot=convert_from_one_hot

    if convert_from_one_hot:
      self.DATA['train_labels']=np.argmax(self.DATA['train_labels'],axis=1)
      self.DATA['valid_labels']=np.argmax(self.DATA['valid_labels'],axis=1)
      self.DATA['test_labels']=np.argmax(self.DATA['test_labels'],axis=1)
      print(self.DATA['train_labels'])

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
    self.tf_batch_dataset = tf.placeholder(dtype=self.dtype,shape=self.data_shape)
    if self.convert_from_one_hot:
      self.tf_batch_labels = tf.placeholder(dtype=self.inttype, shape=(self.batch_size))
    else:
      self.tf_batch_labels = tf.placeholder(dtype=self.inttype, shape=(self.batch_size, self.num_classes))

    if calc_train_acc==True:
      self.tf_train_dataset = tf.constant(self.DATA['train_dataset'], dtype=self.dtype)
      self.tf_train_labels = tf.constant(self.DATA['train_labels'],dtype=self.inttype)

    if calc_valid_acc==True:
      self.tf_valid_dataset = tf.constant(self.DATA['valid_dataset'], dtype=self.dtype)
      self.tf_valid_labels = tf.constant(self.DATA['valid_labels'],dtype=self.inttype)

    if calc_test_acc==True:
      self.tf_test_dataset = tf.constant(self.DATA['test_dataset'], dtype=self.dtype)
      self.tf_test_labels = tf.constant(self.DATA['test_labels'], dtype=self.inttype)

    with tf.name_scope('batch_image'):
      mean = tf.reduce_mean(self.tf_batch_dataset)
      stddev = tf.sqrt(tf.reduce_mean(tf.square(self.tf_batch_dataset - mean)))
      tf.summary.scalar('mean', mean)
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max_abs', tf.reduce_max(self.tf_batch_dataset))
      tf.summary.scalar('min_abs', tf.reduce_min(self.tf_batch_dataset))

    if calc_valid_acc==True:
      with tf.name_scope('valid_image'):
        mean = tf.reduce_mean(self.tf_valid_dataset)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(self.tf_valid_dataset - mean)))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max_abs', tf.reduce_max(self.tf_valid_dataset))
        tf.summary.scalar('min_abs', tf.reduce_min(self.tf_valid_dataset))

  def create_batch(self):
    if len(self.ind)<self.batch_size:
      self.ind.extend(random.sample(range(self.num_train_ex), self.num_train_ex))

    if self.dim==2:
      batch_dataset=self.DATA['train_dataset'][self.ind[0:self.batch_size],:]
    else:
      batch_dataset=self.DATA['train_dataset'][self.ind[0:self.batch_size],:,:,:]

    if len(self.DATA['train_labels'].shape)>1:
      batch_labels=self.DATA['train_labels'][self.ind[0:self.batch_size],:]
    else:
      batch_labels=self.DATA['train_labels'][self.ind[0:self.batch_size]]

    self.ind=self.ind[self.batch_size:]

    return batch_dataset, batch_labels

  def get_next_batch(self):
    batch_dataset, batch_labels = self.create_batch()
    return {self.tf_batch_dataset: batch_dataset, self.tf_batch_labels: batch_labels}
