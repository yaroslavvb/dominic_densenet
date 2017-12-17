"""
==========================================================
Written by Dominic Masters, Graphcore, dominic.masters@graphcore.ai
==========================================================
"""

import tensorflow as tf
import numpy as np

def _variable_on_cpu(name, shape, initializer, dtype=tf.float32, master_weight_type=None, trainable=True):
  """Helper to create a Variable stored on CPU memory.  """
  # with tf.device('/cpu:0'):
  # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  # dtype = tf.float32
  if not master_weight_type==None:
    var = tf.get_variable(name, shape, initializer=initializer, dtype=master_weight_type, trainable=trainable)
    var = tf.cast(var, dtype=dtype)
  else:
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, stddev = None, wd = None, scaling = 1, initializer = 'Normal', dtype = tf.float32, master_weight_type=None):
  """Helper to create an initialized Variable with weight decay. """
  if not master_weight_type==None:
    init_dtype = master_weight_type
  else:
    init_dtype = dtype

  if initializer == 'Normal':
    if stddev == None:
      if len(shape)==4:
        n = np.prod(shape[0:2]) * shape[3]
      else:
        n = np.prod(shape)
      stddev = np.sqrt(2.0 / n)
    initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=init_dtype)
  elif initializer == 'Deep_Rect':
    # initializer from [Delving Deep into Rectifiers](http://arxiv.org/pdf/1502.01852v1.pdf)
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, dtype=init_dtype)
  elif initializer == 'Xavier':
    initializer = tf.contrib.layers.xavier_initializer(uniform=True, dtype=init_dtype)
  elif initializer == 'Uniform':
    initializer = tf.uniform_unit_scaling_initializer(factor=scaling, dtype=init_dtype)

  if not master_weight_type==None:
    var = _variable_on_cpu(name, shape, initializer, dtype=master_weight_type)
    var = tf.cast(var, dtype=dtype)
  else:
    var = _variable_on_cpu(name, shape, initializer, dtype=dtype)

  if wd is not (None or 0):
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    if tf.get_variable_scope().reuse == False:
      tf.add_to_collection('losses', weight_decay)
  return var

def variable_summaries(var,name='summaries', collections=["VARIABLE_SUMMARIES"], stats=['mean','stddev','max','min','hist','log_hist']):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(name):
    if ('mean' in stats) or ('stddev' in stats):
      mean = tf.reduce_mean(var)
    if 'mean' in stats: tf.summary.scalar('mean', mean, collections=collections)
    if 'stddev' in stats:
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev, collections=collections)
    if 'max' in stats: tf.summary.scalar('max', tf.reduce_max(var), collections=collections)
    if 'min' in stats: tf.summary.scalar('min', tf.reduce_min(var), collections=collections)
    # tf.summary.scalar('sparsity', tf.nn.zero_fraction(var), collections=collections)
    if 'hist' in stats: tf.summary.histogram('histogram', var, collections=collections)
    if 'log_hist' in stats:
      tf.summary.histogram('log10_abs',
        tf.maximum(tf.divide(tf.log(tf.abs(tf.cast(var,tf.float32))),tf.log(10.0)),-14.0), collections=collections)

def kernel_to_image(k_temp,layer1=False):
  if layer1==True:
    n=int(np.ceil(np.sqrt(k_temp.get_shape().as_list()[3])))
    d=(n*n)-k_temp.get_shape().as_list()[3]
    k_temp=tf.pad(k_temp,[[1, 1], [1, 1],[0,0],[0,int(d)]],"CONSTANT")
    k_temp=tf.reshape(k_temp,[k_temp.get_shape().as_list()[0],k_temp.get_shape().as_list()[1],k_temp.get_shape().as_list()[2],n,n])
    k_temp=[k_temp[:,:,:,:,j] for j in range(n)]
    k_temp=tf.concat(k_temp,axis=1)
    k_temp=[k_temp[:,:,:,j] for j in range(n)]
    k_temp=tf.concat(k_temp,axis=0)
    k_temp=tf.expand_dims(k_temp,axis=0)
  else:
    k_temp=tf.pad(k_temp,[[1, 1], [1, 1],[0,0],[0,0]],"CONSTANT")
    k_temp=[k_temp[:,:,j,:] for j in range(k_temp.get_shape().as_list()[2])]
    k_temp=tf.concat(k_temp,axis=0)
    k_temp=[k_temp[:,:,j] for j in range(k_temp.get_shape().as_list()[2])]
    k_temp=tf.concat(k_temp,axis=1)
    k_temp=tf.expand_dims(k_temp,axis=2)
    k_temp=tf.expand_dims(k_temp,axis=0)
  return k_temp
