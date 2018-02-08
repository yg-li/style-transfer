import os
import sys
import argparse
import time
import logging

from PIL import Image

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn

import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as utils
import torchvision.datasets as datasets

# run on GPU
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Mean and Standard deviation of the Imagenet dataset
MEAN_IMAGE = [0.485, 0.456, 0.406]
STD_IMAGE = [0.229, 0.224, 0.225]

TV_WEIGHT = 1e-6

# helpers
toTensor = transforms.ToTensor()
normaliseImage = transforms.Normalize(MEAN_IMAGE, STD_IMAGE)

def check_paths(args):
  """ Check if the path exist, if not create one """
  try:
    if not os.path.exists(args.save_model_dir):
      os.makedirs(args.save_model_dir)
    if not os.path.exists(args.checkpoint_dir):
      os.makedirs(args.checkpoint_dir)
  except OSError as e:
    print(e, flush=True)
    sys.exit(1)

def image_loader(image_name, size=None, transform=None):
  ''' helper for loading images
  Args:
    image_name: the path to the image
    size: the size of the loaded image needed to be resize to, can be a sequence or int
  Returns:
    image: the Tensor representing the image loaded (value in [0,1])
  '''
  image = Image.open(image_name)
  if transform is not None:
    image = toTensor(transform(image))
  elif size is not None:
    cutImage = transforms.Resize(size)
    image = toTensor(cutImage(image))
  else:
    image = toTensor(image)
  return image.type(dtype)

def normalize_images(images):
  """ Normalised a batch of images wrt the Imagenet dataset """
  # normalize using imagenet mean and std
  mean = images.data.new(images.data.size())
  std = images.data.new(images.data.size())
  for i in range(3):
    mean[:, i, :, :] = MEAN_IMAGE[i]
    std[:, i, :, :] = STD_IMAGE[i]
  return (images - Variable(mean)) / Variable(std)

class VGG(nn.Module):
  '''
    Module based on pre-trained VGG 19 for extracting high level features of image.
    Use relu3_1 for style-swapping and loss calculation.
  '''
  def __init__(self):
    super(VGG, self).__init__()
    vgg = models.vgg19(pretrained=True).features
    self.slice = nn.Sequential()
    for x in range(12):
      self.slice.add_module(str(x), vgg[x])
    for param in self.parameters():
      param.requires_grad = False

  def forward(self, x):
    return self.slice(x)

# class StyleSwapNet(nn.Module):
#   '''
#     Module doing the style-swapping of content and style activations.
#   '''
#   def __init__(self, style_activation):
#     super(StyleSwapNet, self).__init__()
#     b, ch, h, w = style_activation.size()
#     self.conv = nn.Conv3d(ch, (h-args.patch_size+1)*(w-args.patch_size+1), kernel_size=(ch, args.patch_size, args.patch_size))
#     self.deconv = nn.ConvTranspose3d((h-args.patch_size+1)*(w-args.patch_size+1), ch,
#                                         kernel_size=(ch, args.patch_size, args.patch_size), padding=1)
#     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! should have everything in CUDA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     if use_cuda:
#       self.conv.cuda()
#       self.deconv.cuda()
#     for param in self.parameters():
#       param.requires_grad = False
#
#   def forward(self, x):
#     x = self.conv(x)
#     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! argmax !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     x = self.deconv(x)
#     return x

class InverseNet(nn.Module):
  '''
    Module for approximating the input of VGG19 given its activation at relu3_1.
    The inverse is neither injective nor surjective.
  '''
  def __init__(self):
    super(InverseNet, self).__init__()
    self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
    self.in1 = nn.InstanceNorm2d(128)
    self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.in2 = nn.InstanceNorm2d(128)
    self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
    self.in3 = nn.InstanceNorm2d(64)
    self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.in4 = nn.InstanceNorm2d(64)
    self.conv5 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    self.relu = nn.ReLU()
    self.up = nn.Upsample(scale_factor=2)

  def forward(self, x):
    x = self.relu(self.in1(self.conv1(x)))
    x = self.up(x)
    x = self.relu(self.in2(self.conv2(x)))
    x = self.relu(self.in3(self.conv3(x)))
    x = self.up(x)
    x = self.relu(self.in4(self.conv4(x)))
    x = self.conv5(x)
    return x

def style_swap(content_activation, style_activation):
  _, ch_c, h_c, w_c = content_activation.size()
  _, ch_s, h_s, w_s = style_activation.size()
  if ch_c != ch_s:
    print("ERROR: the layer for content activation and style activation should be the same", flush=True)
    sys.exit(1)

  style_patches_conv = nn.Conv3d(1, (h_s-args.patch_size+1)*(w_s-args.patch_size+1),
                                 kernel_size=(ch_s, args.patch_size, args.patch_size))
  for h in range((int) (args.patch_size/2), h_s-(int) (args.patch_size+1/2)):
    for w in range((int) (args.patch_size/2), w_s-(int) (args.patch_size+1/2)):

  content_activation.data = content_activation.data.unsqueeze_(0)
  for h in range((int) (args.patch_size/2), h_c-(int) (args.patch_size+1/2)):
    for w in range((int) (args.patch_size/2), w_c-(int) (args.patch_size+1/2)):


def stylize(args):
  print('Start stylizing', flush=True)
  content_image = image_loader(args.content_image)
  content_image = content_image.unsqueeze(0)
  content_image = Variable(content_image, volatile=True)
  style_image = image_loader(args.style_image)
  style_image = style_image.unsqueeze(0)
  style_image = Variable(style_image, volatile=True)

  inverse_net = InverseNet()
  inverse_net.load_state_dict(torch.load(args.inverse_net))
  vgg = VGG()
  if use_cuda:
    inverse_net.cuda()
    vgg.cuda()

  content_activation = vgg(content_image)
  style_activation = vgg(style_image)
  target_activation = style_swap(content_activation, style_activation)

  output = inverse_net(target_activation)
  utils.save_image(output.data[0], args.output_image)
  print('Done stylization to', args.output_image, '\n', flush=True)

def main(args):
  if args.subcommand is None:
    print('ERROR: specify either train or eval', flush=True)
    sys.exit(1)

  if args.subcommand == 'train':
    check_paths(args)
    if 'inverse_net.model' in os.listdir(args.save_model_dir):
      print('Already trained the inverse net\n', flush=True)
      return
    logging.basicConfig(filename=args.checkpoint_dir + '/log', level=logging.INFO)
    logging.info('Patch size: ' +  str(args.patch_size))
    train(args)
  else:
    stylize(args)
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest='subcommand')

  train_parser = subparsers.add_parser('train', help='parser for training parameters')
  train_parser.add_argument('--content-dataset', type=str, required=True,
                                help='path to content dataset')
  train_parser.add_argument('--style-dataset', type=str, required=True,
                                help='paths to style dataset')
  train_parser.add_argument('--save-model-dir', type=str, required=True,
                                help='path to folder where trained model will be saved.')
  train_parser.add_argument('--checkpoint-dir', type=str, required=True,
                                help='path to folder where checkpoints of trained models will be saved')
  train_parser.add_argument('--image-size', type=int, default=256,
                                help='size of training images, default is 256 X 256')
  train_parser.add_argument('--epochs', type=int, default=2,
                                help='number of training epochs, default is 2')
  train_parser.add_argument('--batch-size', type=int, default=4,
                                help='batch size for training, default is 4, has to be square of a natural number')
  train_parser.add_argument('--patch-size', type=int, default=3,
                                help='size of patches for activations, default is 3')
  train_parser.add_argument('--lr', type=float, default=1e-3,
                                help='learning rate, default is 1e-3')
  train_parser.add_argument('--log-interval', type=int, default=500,
                                help='number of images after which the training loss is logged, default is 500')
  train_parser.add_argument('--checkpoint-interval', type=int, default=2000,
                                help='number of batches after which a checkpoint of the trained model will be created')

  eval_parser = subparsers.add_parser('eval', help='parser for stylizing arguments')
  eval_parser.add_argument('--content-image', type=str, required=True,
                               help='path to the content image')
  eval_parser.add_argument('--style-image', type=str, required=True,
                           help='path to the style image')
  eval_parser.add_argument('--output-image', type=str, required=True,
                               help='path for saving the output image')
  eval_parser.add_argument('--inverse-net', type=str, required=True,
                               help='saved inverse net to be used for stylizing the image')

  args = parser.parse_args()

  main(args)