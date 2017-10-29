from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.contrib import slim

from nets import vgg
from preprocessing import vgg_preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('content_image', help='name of content image', type=str)
parser.add_argument('style_image', help='name of style image', type=str)
parser.add_argument('-o', '--output_image', help='name of output (stylised) image', type=str)

# The mean need to be subtracted for use with the VGG, as the model was trained this way
RGB_MEAN = [123.68, 116.779, 103.939]

# The layers used for representing contents
content_layer = 'conv4_2'
CONTENT_WEIGHT = 0.01

# The layers used for representing styles
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv_4_1', 'conv5_1']
style_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
STYLE_WEIGHT = 1

def content_loss(p, f):
  """ Compute the content loss, using outputs of conv4_2
  Args:
    p: output of conv4_2 for the content image
    f: output of conv4_2 for the output image

  Returns:
    the content loss
  """
  return 0.5 * tf.reduce_sum(tf.square(f - p))

def gram_matrix(f, N, M):
  """ Generate the gram matrix for output of a layer
  Args:
     f: the output of a convolutional layer
     N: the number of filters
     M: (width * height) of the layer
  """
  f = tf.reshape(f, (N, M))
  return tf.reduce_sum(tf.matmul(f, f, transpose_b=True))

def single_layer_style_loss(a, g):
  """ Compute the style loss for a layer
  Args:
    a: output of a layer for the style image
    g: output of the same layer for the output image

  Returns:
    the style loss for a single layer
  """
  N = a.shape[3] # the number of filters
  M = a.shape[1] * a.shape[2] # (height * width) of that layer
  _A = gram_matrix(a, N, M)
  _G = gram_matrix(g, N, M)
  return tf.reduce_sum(tf.square(_G - _A)) / (4 * (N ** 2) * (M ** 2))

def style_loss(A, G, nets):
  """ Compute the total style loss
  Args:
    A: the output of the layers representing styles of style images
    G: the output of the layers representing styles of output image
    nets: the weights of pre-trained CNN

  returns:
    the total style loss
  """
  E = [single_layer_style_loss(A['vgg_19/'+style_layers[i]], G['vgg_19/'+style_layers[i]])
       for i in range(len(style_layers))]
  return tf.reduce_sum(tf.multiply(E, style_weights))

def total_loss(C, S, O, nets):
  """ Computer the total loss
  Args:
    C: the output of content image in VGG
    S: the output of style image in VGG
    O: the output of output image in VGG
    nets: model variables

  """
  return


def main(args):
  content_image_dir = os.getcwd() + '/images/content_images/' + args.content_image
  style_image_dir = os.getcwd() + '/images/style_images/' + args.style_image
  if not args.output_image is None:
    output_image_dir = os.getcwd() + '/images/output_images/' + args.output_image
  else:
    output_image_dir = os.getcwd() + '/images/output_images/' + args.content_image.split('.')[0] + '_' + args.style_image
  image_names = [content_image_dir, style_image_dir]

  with tf.Graph().as_default():
    """ Should make each image an end_points """
    # Preprocess Content Image
    content_file = tf.read_file(content_image_dir)
    content_image = tf.image.decode_jpeg(content_file, channels=3)
    content_image = tf.subtract(tf.to_float(tf.expand_dims(content_image, 0)), RGB_MEAN)
    with slim.arg_scope(vgg.vgg_arg_scope()):
      nets, content_rep = vgg.vgg_19(content_image, is_training=False)

    # # Preprocess Style Image
    # style_file = tf.read_file(style_image_dir)
    # style_image = tf.image.decode_jpeg(style_file, channels=3)
    # style_image = tf.subtract(tf.to_float(tf.expand_dims(style_image, 0)), RGB_MEAN)
    # with slim.arg_scope(vgg.vgg_arg_scope()):
    #   _, style_rep = vgg.vgg_19(style_image, is_training=False)
    #
    # # Prepare output image
    # output_image = tf.random_uniform((1, content_image.shape[1], content_image.shape[2], 3), -20, 20)
    # with slim.arg_scope(vgg.vgg_arg_scope()):
    #   _, output_rep = vgg.vgg_19(output_image, is_training=False)

if __name__ == "__main__":
  args = parser.parse_args()
  main(args)

  # imagename_queue = tf.train.string_input_producer(image_names, shuffle=False)
  # image_reader = tf.WholeFileReader()
  # _, images_string = image_reader.read(imagename_queue)
  # image = tf.random_crop(tf.image.decode_jpeg(images_string, channels=3), size=[])
  # images = tf.train.batch([image], batch_size=2)