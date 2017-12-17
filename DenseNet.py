"""
==========================================================
Written by Dominic Masters, Graphcore, dominic.masters@graphcore.ai
==========================================================
"""


import re
import sys
import tensorflow as tf
import Tools as N

def preset(case):
  if 'CIFAR' in case:
    input_maxpool = False

    if '100' in case:
      N=(100-4)//6
      block_rep=[N, N, N]
      bottleneck=True
      growth_rate=12
    elif '250' in case:
      N=(250-4)//6
      block_rep=[N, N, N]
      bottleneck=True
      growth_rate=24
    elif '190' in case:
      N=(190-4)//6
      block_rep=[N, N, N]
      bottleneck=True
      growth_rate=40
  else:
    input_maxpool = True

    if '121' in case:
      block_rep=[6, 12, 24, 16]
      bottleneck=True
      growth_rate=32
    elif '169' in case:
      block_rep=[6, 12, 32, 32]
      bottleneck=True
      growth_rate=32
    elif '201' in case:
      block_rep=[6, 12, 48, 32]
      bottleneck=True
      growth_rate=32
    elif '161' in case:
      block_rep=[6, 12, 36, 24]
      bottleneck=True
      growth_rate=48
    else:
      raise "Preset number of layers not available"

  return block_rep, bottleneck, growth_rate, input_maxpool

def inference(tf_dataset,
        NUM_CLASSES,
        training=True,
        reuse=False,
        block_rep=[6,12,24,16],
        growth_rate=32,
        bottleneck=False,
        theta=0.5,
        reg=1e-4,
        batch_norm=True,
        input_maxpool=True,
        dropout_prob=1,
        decay=0.99,
        preset_config=None):
  """Tensorflow implementation of DenseNet as described in:
  Densely Connected Convolutional Networks
  Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten,
  https://arxiv.org/abs/1608.06993

  Params:
  tf_dataset: dataset tensor of shape [batch, height, width, depth] (required)
  NUM_CLASSES: number of class labels, dictates dimensions of output (required)
  training: True if training (required)
  reuse: reuse previous trainable variables, typically False if training (required)
  block_rep: length describes number of dense blocks, values represent number of repetations of each
  growth_rate: growth rate of filters, as described in paper
  bottleneck: Use bottleneck block structure
  theta=0.5: compression rate, as described in paper
  reg=1e-4: weight decay
  batch_norm: use batch norm
  input_maxpool: downsample of first convolution
  dropout_prob: probability of keeping neuron
  decay: batch norm weighted average decay factor
  preset_config: preset configurations for networks described in paper,
      this will overwrite block_rep, bottleneck, growth_rate and input_maxpool

  Written by Dominic Masters, Graphcore, dominic.masters@graphcore.ai

  """

  #----------------------

  if not preset_config == None:
    block_rep, bottleneck, growth_rate, input_maxpool = preset(preset_config)

  batch_size=tf_dataset.get_shape()[0].value
  img_size=[tf_dataset.get_shape()[1].value, tf_dataset.get_shape()[2].value]
  channels=tf_dataset.get_shape()[3]

  #----------------------
  if training and (sys.version_info < (3, 0)):
    print('\nDenseNet_inference parameters:')
    for k, v in list(locals().iteritems()):
      if not k=='tf_dataset' and not k=='labels':
        print(k + ' = ' + str(v))

  TENSOR_IN=tf_dataset
  if training: print('Dataset:',TENSOR_IN)

  def _conv(INPUT, k_size, filters, downsample=False, im_sum=False):
    stride = [1,2,2,1] if downsample else [1,1,1,1]
    W = N._variable_with_weight_decay('weights', shape=[k_size, k_size, INPUT.get_shape()[3].value, filters], initializer = 'Deep_Rect', wd=reg)
    conv = tf.nn.conv2d(INPUT, W, strides=stride, padding='SAME', name='conv2d')
    b = N._variable_on_cpu('biases', [filters], tf.constant_initializer(0.0))
    OUTPUT = tf.nn.bias_add(conv, b)
    return OUTPUT

  def _conv_block(INPUT, k_size, filters, batch_norm=True,  name='conv_block', im_sum=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
      INPUT = tf.identity(INPUT,name='Start_of_' + name)
      if batch_norm:
        INPUT = tf.contrib.layers.batch_norm(INPUT, center=True, scale=True, is_training=training,
            reuse=reuse, decay=decay, scope='BN')

      INPUT = tf.nn.relu(INPUT, name='relu')
      OUTPUT = _conv(INPUT, k_size, filters, im_sum=False)
      if training and (not dropout_prob==1): OUTPUT = tf.nn.dropout(OUTPUT, dropout_prob)
      return OUTPUT

  def _dense_block(INPUT, block_rep, bottleneck, batch_norm,  name='dense_block'):
    with tf.variable_scope(name, reuse=reuse) as scope:
      T = tf.identity(INPUT,name='Start_of_main_block')
      residual = [T]
      for i in range(block_rep):
        # if training: print(name +'_input' + str(i) + ':', T)
        if bottleneck:
          T = tf.identity(T,name='Start_of_' + name)
          T = _conv_block(T, 1, 4*growth_rate, batch_norm=batch_norm, name='conv_block_' + str(i) + 'a')
          # print(name +'_1x1_output' + str(i) + ':', T)
          T = _conv_block(T, 3, growth_rate, batch_norm=batch_norm, name='conv_block_' + str(i) + 'b')
          # print(name +'_3x3_output' + str(i) + ':', T)
        else:
          T = _conv_block(T, 3, growth_rate, batch_norm=batch_norm, name='conv_block_' + str(i))
        T = tf.identity(T,name='End_of_' + name)

        residual.append(T)
        T = tf.concat(residual,3)
        # if training: print(name +'_output' + str(i) + ':', T)
    return T

  def _transition_layer(INPUT, name='transition'):
    """
    Transition layer:
    theta is a compression rate where 0 < theta <= 1
    """
    with tf.variable_scope(name, reuse=reuse) as scope:
      filters = theta * INPUT.get_shape()[3].value
      conv = _conv(INPUT, 1, filters)
      avpool = tf.nn.avg_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='avpool')
    return avpool


  def _fc_block(INPUT, shape, act=True):

    W = N._variable_with_weight_decay('weights', shape=shape, initializer = 'Deep_Rect', wd=reg)
    b = N._variable_on_cpu('biases', [shape[1]],tf.constant_initializer(0.1))
    fc = tf.nn.xw_plus_b(INPUT, W, b, name='fc')
    OUTPUT = tf.nn.relu(fc, name='relu') if act else fc

    if training:
      print('FC_input:', INPUT)
      print('FC_output:', OUTPUT)
    return OUTPUT

  with tf.variable_scope('conv1', reuse=reuse) as scope:

    if input_maxpool:
      TENSOR_IN = _conv(TENSOR_IN, 7, 2*growth_rate, downsample=input_maxpool, im_sum=True)
      TENSOR_IN = tf.nn.max_pool(TENSOR_IN, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')
    else:
      TENSOR_IN = _conv(TENSOR_IN, 3, 2*growth_rate, downsample=input_maxpool, im_sum=True)
    if training: print('conv1_output:',TENSOR_IN)

  for j in range(len(block_rep)):
    if j != 0:
      TENSOR_IN = _transition_layer(TENSOR_IN, name='Transition_' + str(j))
      if training: print('Transition_' + str(j) + '_output:',TENSOR_IN)
    TENSOR_IN = _dense_block(TENSOR_IN,  block_rep[j], bottleneck=bottleneck, batch_norm=batch_norm, name='Dense_Block_' + str(j+1))
    if training: print('Dense_Block_' + str(j+1) + '_output:',TENSOR_IN, '\n')


  with tf.variable_scope('av_FC',reuse=reuse) as scope:
    shape=TENSOR_IN.get_shape().as_list()
    #global ave pool
    avpool = tf.nn.avg_pool(TENSOR_IN, ksize=[1, shape[1], shape[2], 1], strides=[1, 1, 1, 1], padding='VALID', name='avpool')
    if training: print('avpool_output:', avpool)
    reshape = tf.reshape(avpool, [batch_size, -1])
    dim = reshape.get_shape()[1].value

    logits = _fc_block(reshape, [dim, NUM_CLASSES], act=False)

  return logits
