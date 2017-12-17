"""
==========================================================
Written by Dominic Masters, Graphcore, dominic.masters@graphcore.ai
==========================================================
"""

import tensorflow as tf

def pad_flip_and_random_crop(images, padding):
  shape = images.get_shape().as_list()

  def prep_data_augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.pad(image, [[padding,padding],[padding,padding],[0,0]], "CONSTANT")
    image = tf.random_crop(image,shape[1:])
    return image

  return tf.map_fn(prep_data_augment, images)

