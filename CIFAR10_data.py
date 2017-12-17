"""
==========================================================
Written by Dominic Masters, Graphcore, dominic.masters@graphcore.ai
==========================================================
"""

import numpy as np
from six.moves import cPickle as pickle
import os
import sys
import tarfile
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import sys
#import TF_IM_CLASS
import inspect
from six.moves import urllib

mod_dir='/tmp'
image_size = 32
num_classes=10

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

data_dir=mod_dir + '/Datasets/cifar/'
folder_name = 'cifar-10-batches-py'

def unpickle(file):
  with open(file, 'rb') as fo:
    if sys.version_info >= (3, 0):
      dict = pickle.load(fo, encoding='bytes')
    else:
      dict = pickle.load(fo)
  return dict

def _randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:,:]
  shuffled_labels = labels[permutation,:]
  return shuffled_dataset, shuffled_labels

def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  # dest_directory = FLAGS.data_dir
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(data_dir, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(data_dir, folder_name)
  if not os.path.exists(extracted_dir_path):
      tarfile.open(filepath, 'r:gz').extractall(data_dir)

def create_datasets(num_classes=10,train_size = 40000,valid_size = 10000,test_size = 10000, normalise=True, compact=False, randomize=True, shape='2D',
  test_as_valid=False, normalisation_method='simple', dtype=np.float32, inttype=np.int32):

  maybe_download_and_extract()

  full_dataset=np.zeros([50000, image_size, image_size, 3])
  full_labels=np.zeros([50000, num_classes])
  for i in range(5):
    data=unpickle(data_dir + folder_name + '/data_batch_' + str(i+1))
    images=data[b'data'].astype(dtype)
    images=np.reshape(images[:,:],[-1,32,32,3],'F')
    images=np.transpose(images,[0,2,1,3])
    full_dataset[10000*(i):10000*(i+1),:,:,:]=images
    full_labels[10000*(i):10000*(i+1),:]=np.eye(10)[data[b'labels']].astype(inttype)

  if normalise:
    if normalisation_method=='simple':
      full_dataset=full_dataset/255
      full_dataset=full_dataset-0.5
    elif normalisation_method=='per_pixel':
      full_dataset=full_dataset/255
      full_dataset=full_dataset-np.mean(full_dataset, axis=0)
    elif normalisation_method=='full':
      data_mean = np.mean(full_dataset, axis=(0, 3), keepdims=True)
      data_std = np.std(full_dataset, axis=(0, 3), keepdims=True)
      full_dataset=(full_dataset-data_mean)/data_std

  if randomize:
    full_dataset, full_labels = _randomize(full_dataset, full_labels)

  train_dataset=full_dataset[0:train_size,:,:,:]
  train_labels=full_labels[0:train_size,:]

  valid_dataset=full_dataset[train_size:(train_size+valid_size),:,:,:]
  valid_labels=full_labels[train_size:(train_size+valid_size),:]

  data=unpickle(data_dir + folder_name + '/test_batch')
  images=data[b'data'].astype(dtype)
  images=np.reshape(images[:,:],[-1,32,32,3],'F')
  images=np.transpose(images,[0,2,1,3])
  test_dataset=images[0:test_size,:,:,:]
  test_labels=np.eye(10)[data[b'labels']].astype(inttype)
  test_labels=test_labels[0:test_size,:]

  if normalise:
    if normalisation_method=='simple':
      test_dataset=test_dataset/255
      test_dataset=test_dataset-0.5
    elif normalisation_method=='per_pixel':
      test_dataset=test_dataset/255
      test_dataset=test_dataset-np.mean(test_dataset, axis=0)
    elif normalisation_method=='full':
      test_dataset=(test_dataset-data_mean)/data_std
    # test_dataset=(test_dataset/255)-0.5

  if test_as_valid:
    valid_dataset=test_dataset
    valid_labels=test_labels
    # test_dataset=[]
    # test_labels=[]

  print('Training:', train_dataset.shape, train_labels.shape)
  print('Validation:', valid_dataset.shape, valid_labels.shape)
  print('Testing:', test_dataset.shape, test_labels.shape)

  if shape=='2D':
    train_dataset = train_dataset.reshape((-1, image_size * image_size * 3)).astype(dtype)
    valid_dataset = valid_dataset.reshape((-1, image_size * image_size * 3)).astype(dtype)
    test_dataset = test_dataset.reshape((-1, image_size * image_size * 3)).astype(dtype)


  if compact:
    DATA={'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels}
    return DATA
  else:  return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

