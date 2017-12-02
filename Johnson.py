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
    if not os.path.exists(args.checkpoint_dir):
      os.makedirs(args.checkpoint_dir)
  except OSError as e:
    print(e, flush=True)
    sys.exit(1)

def image_loader(image_name, scale=None, transform=None):
  ''' helper for loading images
  Args:
    image_name: the path to the image
    height, width: the height and width the loaded image needed to be resized to
  Returns:
    image: the Tensor representing the image loaded (value in [0,1])
  '''
  image = Image.open(image_name)
  if transform is not None:
    image = toTensor(transform(image))
  elif scale is not None:
    cutImage = transforms.Scale(scale)
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

class LossNetwork(nn.Module):
  ''' Module based on pre-trained VGG 19 for extracting high level features of image
    Use relu3_3 for content representation
    Use relu1_2, relu2_2, relu3_3, and relu4_3 for style representation
  '''
  def __init__(self):
    super(LossNetwork, self).__init__()
    vgg = models.vgg16(pretrained=True).features
    self.slice1 = nn.Sequential()
    self.slice2 = nn.Sequential()
    self.slice3 = nn.Sequential()
    self.slice4 = nn.Sequential()
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
    relu1_2 = self.slice1(x)
    relu2_2 = self.slice2(relu1_2)
    relu3_3 = self.slice3(relu2_2)
    relu4_3 = self.slice4(relu3_3)
    outputs = namedtuple('Outputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
    return outputs(relu1_2, relu2_2, relu3_3, relu4_3)

def gram_matrix(input):
  """ Compute batch-wise gram matrices """
  b, ch, h, w = input.size()
  features = input.view(b, ch, h * w)
  G = features.bmm(features.transpose(1,2))
  return G / (h * w)

class TransformerNetwork(nn.Module):
  ''' Module implementing the image transformation network '''
  def __init__(self):
    super(TransformerNetwork, self).__init__()
    # Downsampling
    self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1)
    self.in1 = nn.InstanceNorm2d(32, affine=True)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
    self.in2 = nn.InstanceNorm2d(64, affine=True)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
    self.in3 = nn.InstanceNorm2d(128, affine=True)
    # Residual layers
    self.res1 = ResidualBlock(128)
    self.res2 = ResidualBlock(128)
    self.res3 = ResidualBlock(128)
    self.res4 = ResidualBlock(128)
    self.res5 = ResidualBlock(128)
    # Upsampling
    self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding = 1)
    self.in4 = nn.InstanceNorm2d(64, affine=True)
    self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding = 1)
    self.in5 = nn.InstanceNorm2d(32, affine=True)
    self.conv4 = nn.Conv2d(32, 3, kernel_size=9, stride=1)
    self.in6 = nn.InstanceNorm2d(3, affine=True)
    # Non-linearities
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()
    # Reflection paddings
    self.pad_4 = nn.ReflectionPad2d(4)
    self.pad_1 = nn.ReflectionPad2d(1)

  def forward(self, x):
    y = self.relu(self.in1(self.conv1(self.pad_4(x))))
    size1 = y.size()
    y = self.relu(self.in2(self.conv2(self.pad_1(y))))
    size2 = y.size()
    y = self.relu(self.in3(self.conv3(self.pad_1(y))))
    y = self.res1(y)
    y = self.res2(y)
    y = self.res3(y)
    y = self.res4(y)
    y = self.res5(y)
    y = self.relu(self.in4(self.deconv1(y, output_size=size2)))
    y = self.relu(self.in5(self.deconv2(y, output_size=size1)))
    y = self.tanh(self.in6(self.conv4(self.pad_4(y))))
    return y.add(1).div(2)

class ResidualBlock(nn.Module):
  ''' Residual block
      It helps the network to learn the identify function
  '''
  def __init__(self, channels):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1)
    self.in1 = nn.InstanceNorm2d(channels, affine=True)
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1)
    self.in2 = nn.InstanceNorm2d(channels, affine=True)
    self.relu = nn.ReLU()
    self.pad_1 = nn.ReflectionPad2d(1)

  def forward(self, x):
    y = self.relu(self.in1(self.conv1(self.pad_1(x))))
    y = self.in2(self.conv2(self.pad_1(y)))
    return y + x

def train(args):
  print('Start training', flush=True)

  # Training data
  transform = transforms.Compose([
    transforms.Scale(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor()
  ])
  train_dataset = datasets.ImageFolder(args.dataset, transform)
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

  # Networks
  print('Loading networks', flush=True)
  transformer = TransformerNetwork()
  optimizer = Adam(transformer.parameters(), args.lr)
  mse_loss = nn.MSELoss()

  e_has = 0
  batch_has = 0
  checkpoints = os.listdir(args.checkpoint_dir)
  if len(checkpoints) > 1:
    if 'log' in checkpoints:
      checkpoints.remove('log')
    if '.DS_Store' in checkpoints:
      checkpoints.remove('.DS_Store')
    for f in checkpoints:
      if int(f.split('.')[0].split('_')[0]) > e_has:
        e_has = int(f.split('.')[0].split('_')[0])
    for f in checkpoints:
      if int(f.split('.')[0].split('_')[0]) == e_has and int(f.split('.')[0].split('_')[1]) > batch_has:
        batch_has = int(f.split('.')[0].split('_')[1])
    print('e_has:', str(e_has), 'batch_has:', str(batch_has))

    if e_has != 0 or batch_has != 0:
      transformer.load_state_dict(torch.load(args.checkpoint_dir + '/' + str(e_has) + '_' + str(batch_has) + '.pth'))

  vgg = LossNetwork()

  if torch.cuda.device_count() > 1:
    print('Using', torch.cuda.device_count(), 'GPUs', flush=True)
    transformer = nn.DataParallel(transformer)
    vgg = nn.DataParallel(vgg)
  if use_cuda:
    transformer.cuda()
    vgg.cuda()

  # Target of style
  style = Variable(normaliseImage(image_loader(args.style_image, scale=args.image_size)))
  style = style.repeat(args.batch_size, 1, 1, 1)
  features_style = vgg(style)
  target_gram_style = [gram_matrix(x) for x in features_style]

  for e in range(args.epochs):
    if e < e_has:
      print('Epoch', str(e+1), 'has already trained', flush=True)
      continue
    print('Start epoch', str(e+1), flush=True)
    transformer.train()
    agg_content_loss = 0.0
    agg_style_loss = 0.0
    count = 0
    for batch_id, (x, _) in enumerate(train_loader):
      count += len(x)
      if batch_id < batch_has:
        if (batch_id+1) % args.log_interval == 0:
          print('Skipping through batch' , str(batch_id + 1), flush=True)
        continue
      batch_has = 0

      optimizer.zero_grad()
      x = Variable(x).type(dtype)
      y = transformer(x)

      tv_loss = TV_WEIGHT * (torch.sum(torch.abs(y[:,:,:,:-1]-y[:,:,:,1:])) +
                             torch.sum(torch.abs(y[:,:,:-1,:]-y[:,:,1:,:])))

      if (batch_id + 1) % args.log_interval == 0:
        utils.save_image(x.data[0], args.checkpoint_dir + '/' + str(e) + '_' + str(batch_id + 1) + '_content.jpg')
        utils.save_image(y.data[0], args.checkpoint_dir + '/' + str(e) + '_' + str(batch_id + 1) + '_output.jpg')

      x = normalize_images(x)
      y = normalize_images(y)

      features_x = vgg(x)
      features_y = vgg(y)

      content_loss = args.content_weight * mse_loss(features_y.relu3_3, features_x.relu3_3)

      style_loss = 0.
      for ft_y, tg_s in zip(features_y, target_gram_style):
        gm_y = gram_matrix(ft_y)
        style_loss += args.style_weight * mse_loss(gm_y, tg_s[:len(x), :, :])

      total_loss = content_loss + style_loss + tv_loss
      total_loss.backward()
      optimizer.step()

      agg_content_loss += content_loss.data[0]
      agg_style_loss += style_loss.data[0]

      if (batch_id + 1) % args.log_interval == 0:
        mesg = '{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}\tbatch: {}'.format(
                time.ctime(), e + 1, count, len(train_dataset),
                agg_content_loss / args.log_interval,
                agg_style_loss / args.log_interval,
                (agg_content_loss + agg_style_loss) / args.log_interval,
                batch_id + 1)
        logging.info(mesg)
        print(mesg, flush=True)
        agg_content_loss = 0.
        agg_style_loss = 0.

      if (batch_id + 1) % args.checkpoint_interval == 0:
        transformer.eval()
        if use_cuda:
          transformer.cpu()
        ckpt_model_filename = str(e) + '_' + str(batch_id + 1) + '.pth'
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
  save_model_path = os.path.join(args.save_model_dir, save_model_filename)
  torch.save(transformer.state_dict(), save_model_path)

  print('Done, trained model saved at', save_model_path, '\n', flush=True)

def stylize(args):
  print('Start stylizing', flush=True)
  content_image = image_loader(args.content_image)
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

  if args.subcommand == 'train':
    check_paths(args)
    if (args.style_image.split('/')[-1]).split('.')[0] + '.model' in os.listdir(args.save_model_dir):
      print('Already trained for this style\n', flush=True)
      return
    logging.basicConfig(filename=args.checkpoint_dir + '/log', level=logging.INFO)
    logging.info('Content weight: ' +  str(args.content_weight) + ' Style weight: ' + str(args.style_weight))
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
  train_parser.add_argument('--checkpoint-dir', type=str, required=True,
                                help='path to folder where checkpoints of trained models will be saved')
  train_parser.add_argument('--image-size', type=int, default=256,
                                help='size of training images, default is 256 X 256')
  train_parser.add_argument('--epochs', type=int, default=2,
                                help='number of training epochs, default is 2')
  train_parser.add_argument('--batch-size', type=int, default=4,
                                help='batch size for training, default is 4')
  train_parser.add_argument('--content-weight', type=float, default=10,
                                help='weight for content-loss, default is 10')
  train_parser.add_argument('--style-weight', type=float, default=500,
                                help='weight for style-loss, default is 500')
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
