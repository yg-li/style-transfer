import os
import sys
import argparse
import time
import logging
from collections import namedtuple

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
    if args.checkpoint_dir is not None and not (os.path.exists(args.checkpoint_dir)):
      os.makedirs(args.checkpoint_dir)
  except OSError as e:
    print(e, flush=True)
    sys.exit(1)

def image_loader(image_name, height=None, width=None):
  ''' helper for loading images
  Args:
    image_name: the path to the image
    height, width: the height and width the loaded image needed to be resized to
  Returns:
    image: the Tensor representing the image loaded (value in [0,1])
  '''
  image = Image.open(image_name)
  if height is None or width is None:
    image = toTensor(image).type(dtype)
  else:
    image = toTensor(image.resize((width, height))).type(dtype)
  return image

def normalize_images(images):
  """ Normalised a batch of images wrt the Imagenet dataset """
  # normalize using imagenet mean and std
  mean = images.data.new(images.data.size())
  std = images.data.new(images.data.size())
  for i in range(3):
    mean[:, i, :, :] = MEAN_IMAGE[i]
    std[:, i, :, :] = STD_IMAGE[i]
  return (images - Variable(mean)) / Variable(std)

class LossNetwork(torch.nn.Module):
  ''' Module based on pre-trained VGG 19 for extracting high level features of image
    Use relu3_3 for content representation
    Use relu1_2, relu2_2, relu3_3, and relu4_3 for style representation
  '''
  def __init__(self):
    super(LossNetwork, self).__init__()
    vgg = models.vgg16(pretrained=True).features
    self.slice1 = torch.nn.Sequential()
    self.slice2 = torch.nn.Sequential()
    self.slice3 = torch.nn.Sequential()
    self.slice4 = torch.nn.Sequential()
    for x in range(4):
      self.slice1.add_module(str(x), vgg[x])
    for x in range(4, 9):
      self.slice2.add_module(str(x), vgg[x])
    for x in range(9, 16):
      self.slice3.add_module(str(x), vgg[x])
    for x in range(16, 23):
      self.slice4.add_module(str(x), vgg[x])
    for param in self.parameters():
      param.requires_grad = False

  def forward(self, x):
    y = self.slice1(x)
    relu1_2 = y
    y = self.slice2(y)
    relu2_2 = y
    y = self.slice3(y)
    relu3_3 = y
    y = self.slice4(y)
    relu4_3 = y
    outputs = namedtuple('Outputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
    out = outputs(relu1_2, relu2_2, relu3_3, relu4_3)
    return out

def gram_matrix(input):
  """ Compute batch-wise gram matrices """
  (b, ch, h, w) = input.size()
  features = input.view(b, ch, w * h)
  G = features.bmm(features.transpose(1,2))
  return G / (ch * h * w)

class TransformerNetwork(torch.nn.Module):
  ''' Module implementing the image transformation network '''
  def __init__(self):
    super(TransformerNetwork, self).__init__()
    # Downsampling
    self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
    self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
    self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
    self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
    self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
    self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
    # Residual layers
    self.res1 = ResidualBlock(128)
    self.res2 = ResidualBlock(128)
    self.res3 = ResidualBlock(128)
    self.res4 = ResidualBlock(128)
    self.res5 = ResidualBlock(128)
    # Upsampling
    self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
    self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
    self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
    self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
    self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
    # Non-linearities
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    y = self.relu(self.in1(self.conv1(x)))
    y = self.relu(self.in2(self.conv2(y)))
    y = self.relu(self.in3(self.conv3(y)))
    y = self.res1(y)
    y = self.res2(y)
    y = self.res3(y)
    y = self.res4(y)
    y = self.res5(y)
    y = self.relu(self.in4(self.deconv1(y)))
    y = self.relu(self.in5(self.deconv2(y)))
    y = self.deconv3(y)
    return y

class ConvLayer(torch.nn.Module):
  ''' Module representing a convolutional layer which preserves the size of input '''
  def __init__(self, in_channels, out_channels, kernel_size, stride):
    super(ConvLayer, self).__init__()
    reflection_padding = kernel_size // 2
    self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
    self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

  def forward(self, x):
    y = self.reflection_pad(x)
    y = self.conv2d(y)
    return y

class ResidualBlock(torch.nn.Module):
  ''' Residual block
      It helps the network to learn the identify function
  '''
  def __init__(self, channels):
    super(ResidualBlock, self).__init__()
    self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
    self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
    self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
    self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    res = x
    y = self.relu(self.in1(self.conv1(x)))
    y = self.in2(self.conv2(y))
    y = y + res
    return y

class UpsampleConvLayer(torch.nn.Module):
  ''' UpsampleConvLayer
      Upsample the input then do a convolution
  '''
  def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
    super(UpsampleConvLayer, self).__init__()
    self.upsample_layer = torch.nn.Upsample(scale_factor=upsample, mode='nearest')
    reflection_padding = kernel_size // 2
    self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
    self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

  def forward(self, x):
    y = self.upsample_layer(x)
    y = self.reflection_pad(y)
    y = self.conv2d(y)
    return y

def train(args):
  print('Start training', flush=True)
  # np.random.seed(args.seed)
  # torch.manual_seed(args.seed)
  #
  # if use_cuda:
  #   torch.cuda.manual_seed

  # Training data
  transform = transforms.Compose([
    transforms.Scale(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor()
  ])
  train_dataset = datasets.ImageFolder(args.dataset, transform)
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

  style = Variable(normaliseImage(image_loader(args.style_image))).type(dtype)
  style = style.repeat(args.batch_size, 1, 1, 1)

  # Networks
  print('Loading networks', flush=True)
  transformer = TransformerNetwork()
  optimizer = Adam(transformer.parameters(), args.lr)
  mse_loss = torch.nn.MSELoss()

  vgg = LossNetwork()

  if torch.cuda.device_count() > 1:
    print('Using', torch.cuda.device_count(), 'GPUs', flush=True)
    transformer = nn.DataParallel(transformer)
    vgg = nn.DataParallel(vgg)
  if use_cuda:
    transformer.cuda()
    vgg.cuda()

  # Target of style
  features_style = vgg(style)
  target_gram_style = [gram_matrix(x) for x in features_style]

  for e in range(args.epochs):
    print('Start epoch', str(e+1), flush=True)
    transformer.train()
    agg_content_loss = 0.0
    agg_style_loss = 0.0
    count = 0
    for batch_id, (x, _) in enumerate(train_loader):
      size_batch = len(x)
      count += size_batch

      optimizer.zero_grad()
      x = Variable(x).type(dtype)
      y = transformer(x)

      tv_loss = TV_WEIGHT * (torch.sum(torch.abs(y[:,:,:,:-1]-y[:,:,:,1:])) +
                             torch.sum(torch.abs(y[:,:,:-1,:]-y[:,:,1:,:])))

      x = normalize_images(x)
      y = normalize_images(y)

      features_x = vgg(x)
      features_y = vgg(y)

      content_loss = args.content_weight * mse_loss(features_y.relu3_3, features_x.relu3_3)

      style_loss = 0.
      for ft_y, tg_s in zip(features_y, target_gram_style):
        gm_y = gram_matrix(ft_y)
        style_loss += mse_loss(gm_y, tg_s[:size_batch, :, :])
      style_loss *= args.style_weight

      total_loss = content_loss + style_loss + tv_loss
      total_loss.backward()
      optimizer.step()

      agg_content_loss += content_loss.data[0]
      agg_style_loss += style_loss.data[0]

      if (batch_id + 1) % args.log_interval == 0:
        mesg = '{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}'.format(
                time.ctime(), e + 1, count, len(train_dataset),
                agg_content_loss / (batch_id + 1),
                agg_style_loss / (batch_id + 1),
                (agg_content_loss + agg_style_loss) / (batch_id + 1))
        logging.info(mesg)
        print(mesg, flush=True)

      if args.checkpoint_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
        transformer.eval()
        if use_cuda:
          transformer.cpu()
        ckpt_model_filename = 'ckpt_epoch_' + str(e) + '_batch_id_' + str(batch_id + 1) + '.pth'
        ckpt_model_path = os.path.join(args.checkpoint_dir, ckpt_model_filename)
        torch.save(transformer.state_dict(), ckpt_model_path)
        if use_cuda:
          transformer.cuda()
        transformer.train()

  # save model
  transformer.eval()
  if use_cuda:
    transformer.cpu()
  save_model_filename = (args.style_image.split('/')[-1]).split('.')[0] + '.model'
  # save_model_filename = 'epoch_' + str(args.epochs) + '_' + str(time.ctime()).replace(' ', '_') + '_' + str(
  #   args.content_weight) + '_' + str(args.style_weight) + '.model'
  save_model_path = os.path.join(args.save_model_dir, save_model_filename)
  torch.save(transformer.state_dict(), save_model_path)

  print('Done, trained model saved at', save_model_path, '\n', flush=True)

def stylize(args):
  print('Start stylizing', flush=True)
  content_image = image_loader(args.content_image).type(dtype)
  content_image = content_image.unsqueeze(0)
  content_image = Variable(content_image, volatile=True)

  style_model = TransformerNetwork()
  style_model.load_state_dict(torch.load(args.model))
  if use_cuda:
    style_model.cuda()

  output = style_model(content_image)
  utils.save_image(output.data[0], args.output_image)
  print('Done stylization to', args.output_image, '\n', flush=True)

def main(args):
  if args.subcommand is None:
    print('ERROR: specify either train or eval', flush=True)
    sys.exit(1)

  logging.basicConfig(filename=args.checkpoint_dir + '/log', level=logging.DEBUG)

  if args.subcommand == 'train':
    check_paths(args)
    train(args)
  else:
    stylize(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest='subcommand')

  train_parser = subparsers.add_parser('train', help='parser for training parameters')
  train_parser.add_argument('--dataset', type=str, required=True,
                                help='path to training dataset')
  train_parser.add_argument('--style-image', type=str, required=True,
                                help='path to style-image')
  train_parser.add_argument('--save-model-dir', type=str, required=True,
                                help='path to folder where trained model will be saved.')
  train_parser.add_argument('--checkpoint-dir', type=str, default=None,
                                help='path to folder where checkpoints of trained models will be saved')
  train_parser.add_argument('--image-size', type=int, default=256,
                                help='size of training images, default is 256 X 256')
  train_parser.add_argument('--epochs', type=int, default=2,
                                help='number of training epochs, default is 2')
  train_parser.add_argument('--batch-size', type=int, default=4,
                                help='batch size for training, default is 4')
  train_parser.add_argument('--seed', type=int, default=42,
                                help='random seed for training')
  train_parser.add_argument('--content-weight', type=float, default=1e5,
                                help='weight for content-loss, default is 1e5')
  train_parser.add_argument('--style-weight', type=float, default=1e10,
                                help='weight for style-loss, default is 1e10')
  train_parser.add_argument('--lr', type=float, default=1e-3,
                                help='learning rate, default is 1e-3')
  train_parser.add_argument('--log-interval', type=int, default=500,
                                help='number of images after which the training loss is logged, default is 500')
  train_parser.add_argument('--checkpoint-interval', type=int, default=2000,
                                help='number of batches after which a checkpoint of the trained model will be created')

  eval_parser = subparsers.add_parser('eval', help='parser for stylizing arguments')
  eval_parser.add_argument('--content-image', type=str, required=True,
                               help='path to the content image')
  eval_parser.add_argument('--output-image', type=str, required=True,
                               help='path for saving the output image')
  eval_parser.add_argument('--model', type=str, required=True,
                               help='saved model to be used for stylizing the image')

  args = parser.parse_args()

  main(args)
