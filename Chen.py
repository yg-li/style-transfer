import os
import sys
import argparse
import time
import logging
import math
import warnings

from PIL import Image
from PIL import ImageFile

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn

import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as utils
import torchvision.datasets as datasets

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

# run on GPU
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Mean and Standard deviation of the Imagenet dataset
MEAN_IMAGE = [0.485, 0.456, 0.406]
STD_IMAGE = [0.229, 0.224, 0.225]

TV_WEIGHT = 1e-6

# helpers
toTensor = transforms.ToTensor()


def check_paths():
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
  """ Helper for loading images
  Args:
    image_name: the path to the image
    size: the size of the loaded image needed to be resize to, can be a sequence or int
    transform: the transforms need to apply on the loader image
  Returns:
    image: the Tensor representing the image loaded (value in [0,1])
  """
  image = Image.open(image_name)
  if transform is not None:
    image = toTensor(transform(image))
  elif size is not None:
    cut_image = transforms.Resize(size)
    image = toTensor(cut_image(image))
  else:
    image = toTensor(image)
  return image.type(dtype)


def normalize_images(images):
  """ Normalised a batch of images wrt the Imagenet dataset """
  # normalize using imagenet mean and std
  mean = torch.zeros(images.data.size()).type(dtype)
  std = torch.zeros(images.data.size()).type(dtype)
  for i in range(3):
    mean[:, i, :, :] = MEAN_IMAGE[i]
    std[:, i, :, :] = STD_IMAGE[i]
  return (images - Variable(mean, requires_grad=False)) / Variable(std, requires_grad=False)


# def denormalize_image(image):
#   """ Denormalised a batch of images wrt the Imagenet dataset """
#   # denormalize using imagenet mean and std
#   mean = torch.zeros(image.size()).type(dtype)
#   std = torch.zeros(image.size()).type(dtype)
#   for i in range(3):
#     mean[i, :, :] = MEAN_IMAGE[i]
#     std[i, :, :] = STD_IMAGE[i]
#   return (image * std) + mean


class VGG(nn.Module):
  """
    Module based on pre-trained VGG 19 for extracting high level features of image.
    Use relu3_1 for style-swapping and loss calculation.
  """
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


class InverseNet(nn.Module):
  """
    Module for approximating the input of VGG19 given its activation at relu3_1.
    The inverse is neither injective nor surjective.
  """
  def __init__(self):
    super(InverseNet, self).__init__()
    self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
    self.in1 = nn.InstanceNorm2d(128, affine=True)
    self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.in2 = nn.InstanceNorm2d(128, affine=True)
    self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
    self.in3 = nn.InstanceNorm2d(64, affine=True)
    self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.in4 = nn.InstanceNorm2d(64, affine=True)
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

  # Convolutional layer for performing the calculation of cosine measures
  conv = nn.Conv3d(1, (h_s - int(args.patch_size / 2) * 2) * (w_s - int(args.patch_size / 2) * 2),
                   kernel_size=(ch_s, args.patch_size, args.patch_size))
  for param in conv.parameters():
    param.requires_grad = False
  if use_cuda:
    conv.cuda()

  for h in range(h_s - int(args.patch_size / 2) * 2):
    for w in range(w_s - int(args.patch_size / 2) * 2):
      conv.weight[h * (w_s - int(args.patch_size / 2) * 2) + w, 0, :, :, :] = nn.functional.normalize(
        style_activation.data[:, :, h:h + args.patch_size, w:w + args.patch_size])

  # Convolution and taking the maximum of cosine mearsures
  k = conv(content_activation.unsqueeze(0)).squeeze()
  _, max_index = k.max(0)

  # Constructing target activation
  overlaps = torch.zeros(h_c, w_c).type(dtype)
  target_activation = Variable(torch.zeros(content_activation.size()).type(dtype), requires_grad=False)
  for h in range(h_c - int(args.patch_size / 2) * 2):
    for w in range(w_c - int(args.patch_size / 2) * 2):
      s_w = int(max_index.data[h, w] % (w_s - int(args.patch_size / 2) * 2))
      s_h = int((max_index.data[h, w] - s_w) / (w_s - int(args.patch_size / 2) * 2))
      target_activation[:, :, h:h + args.patch_size, w:w + args.patch_size] = \
        target_activation[:, :, h:h + args.patch_size, w:w + args.patch_size] + \
        style_activation[:, :, s_h:s_h + args.patch_size, s_w:s_w + args.patch_size]
      overlaps[h:h + args.patch_size, w:w + args.patch_size].add_(1)
  for h in range(h_c):
    for w in range(w_c):
      target_activation[:, :, h, w] = target_activation[:, :, h, w].div(overlaps[h, w])

  return target_activation


def train():
  print('Start training', flush=True)

  # Training data
  transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor()
  ])
  content_dataset = datasets.ImageFolder(args.content_dataset, transform)
  content_loader = DataLoader(content_dataset, batch_size=int(math.sqrt(args.batch_size)), drop_last=True)
  styel_dataset = datasets.ImageFolder(args.style_dataset, transform)
  style_loader = DataLoader(styel_dataset, batch_size=int(math.sqrt(args.batch_size)), drop_last=True)

  # Networks
  print('Loading networks', flush=True)
  inverse_net = InverseNet()

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
      if int(f.split('.')[0].split('_')[0]) == e_has and \
                      len(f.split('.')[0].split('_')) == 2 and \
                      int(f.split('.')[0].split('_')[1]) > batch_has:
        batch_has = int(f.split('.')[0].split('_')[1])
    batch_has = batch_has // args.checkpoint_interval * args.checkpoint_interval
    print('e_has:', str(e_has), 'batch_has:', str(batch_has))

    if e_has != 0 or batch_has != 0:
      inverse_net.load_state_dict(torch.load(args.checkpoint_dir + '/' + str(e_has) + '_' + str(batch_has) + '.pth'))

  optimizer = Adam(inverse_net.parameters(), args.lr)
  mse_loss = nn.MSELoss()
  vgg = VGG()

  if torch.cuda.device_count() > 1:
    print('Using', torch.cuda.device_count(), 'GPUs', flush=True)
    inverse_net = nn.DataParallel(inverse_net)
    vgg = nn.DataParallel(vgg)
  if use_cuda:
    inverse_net.cuda()
    vgg.cuda()

  # Training epoches
  for e in range(args.epochs):
    if e < e_has:
      print('Epoch', str(e+1), 'has already trained', flush=True)
      continue
    print('Start epoch', str(e+1), flush=True)
    inverse_net.train()
    agg_loss = 0.0
    count = 0
    # Training iteration in one epoch
    for batch_id, ((c, _), (s, _)) in enumerate(zip(content_loader, style_loader)):
      count += len(c)
      if batch_id < batch_has:
        if (batch_id+1) % args.log_interval == 0:
          print('Skipping through batch', str(batch_id + 1), flush=True)
        continue
      batch_has = 0

      if (batch_id + 1) % args.log_interval == 0:
        utils.save_image(c[0], args.checkpoint_dir + '/' + str(e) + '_' + str(batch_id + 1) + '_content.jpg')
        utils.save_image(s[0], args.checkpoint_dir + '/' + str(e) + '_' + str(batch_id + 1) + '_style.jpg')

      c = Variable(c.type(dtype), requires_grad=False)
      s = Variable(s.type(dtype), requires_grad=False)
      c_activations = vgg(normalize_images(c))
      s_activations = vgg(normalize_images(s))
      _, ch, h, w = c_activations.size()
      target_activations = Variable(torch.zeros(args.batch_size, ch, h, w).type(dtype), requires_grad=False)
      for i in range(int(math.sqrt(args.batch_size))):
        for j in range(int(math.sqrt(args.batch_size))):
          # Unsqueeze here since the indexing would remove the first dimention from Variables
          target_activations[i * int(math.sqrt(args.batch_size)) + j] = style_swap(c_activations[i].unsqueeze(0),
                                                                                   s_activations[j].unsqueeze(0))

      optimizer.zero_grad()
      output = inverse_net(target_activations)

      if (batch_id + 1) % args.log_interval == 0:
        utils.save_image(output.data[0], args.checkpoint_dir + '/' + str(e) + '_' + str(batch_id + 1) + '_output.jpg')

      tv_loss = TV_WEIGHT * (torch.sum(torch.abs(output[:, :, :, :-1]-output[:, :, :, 1:])) +
                             torch.sum(torch.abs(output[:, :, :-1, :]-output[:, :, 1:, :])))

      output_activations = vgg(normalize_images(output))
      activation_loss = mse_loss(output_activations, target_activations)

      # print('{}\tBatch {}\tac_loss {}\ttv_loss {}'.format(
      #   time.ctime(), batch_id, activation_loss.data[0], tv_loss.data[0]), flush=True)

      total_loss = activation_loss + tv_loss
      total_loss.backward()
      optimizer.step()

      agg_loss += total_loss.data[0]

      if (batch_id + 1) % args.log_interval == 0:
        mesg = '{}\tEpoch {}:\t[{}/{}]\ttotal: {:.6f}\tbatch: {}'.format(
                time.ctime(), e + 1, count, min(len(content_dataset), len(styel_dataset)),
                agg_loss / args.log_interval,
                batch_id + 1)
        logging.info(mesg)
        print(mesg, flush=True)
        agg_loss = 0

      if (batch_id + 1) % args.checkpoint_interval == 0:
        inverse_net.eval()
        if use_cuda:
          inverse_net.cpu()
        ckpt_model_filename = str(e) + '_' + str(batch_id + 1) + '.pth'
        ckpt_model_path = os.path.join(args.checkpoint_dir, ckpt_model_filename)
        torch.save(inverse_net.state_dict(), ckpt_model_path)
        if use_cuda:
          inverse_net.cuda()
        inverse_net.train()

  # save model
  inverse_net.eval()
  if use_cuda:
    inverse_net.cpu()
  save_model_filename = 'inverse_net.model'
  save_model_path = os.path.join(args.save_model_dir, save_model_filename)
  torch.save(inverse_net.state_dict(), save_model_path)

  print('Done, trained model saved at', save_model_path, '\n', flush=True)


def stylize():
  print('Start stylizing', flush=True)
  content_image = Variable(image_loader(args.content_image).unsqueeze_(0).type(dtype), volatile=True)
  style_image = Variable(image_loader(args.style_image).unsqueeze_(0).type(dtype), volatile=True)

  inverse_net = InverseNet()
  inverse_net.load_state_dict(torch.load(args.inverse_net))
  vgg = VGG()
  if use_cuda:
    inverse_net.cuda()
    vgg.cuda()

  content_activation = vgg(normalize_images(content_image))
  style_activation = vgg(normalize_images(style_image))
  target_activation = style_swap(content_activation, style_activation)

  output = inverse_net(target_activation) #target_activation)
  utils.save_image(output.data[0], args.output_image)
  print('Done stylization to', args.output_image, '\n', flush=True)


def main():
  if args.subcommand is None:
    print('ERROR: specify either train or eval', flush=True)
    sys.exit(1)
  # elif args.patch_size % 2 == 0:
  #   print('ERROR: the patch size must be odd', flush=True)
  #   sys.exit(1)
  # elif math.sqrt(args.batch_size) != int(math.sqrt(args.batch_size)):
  #   print('ERROR: the batch size must be perfect square', flush=True)
  #   sys.exit(1)

  if args.subcommand == 'train':
    check_paths()
    if 'inverse_net.model' in os.listdir(args.save_model_dir):
      print('Already trained the inverse net\n', flush=True)
      return
    logging.basicConfig(filename=args.checkpoint_dir + '/log', level=logging.INFO)
    logging.info('Patch size: ' + str(args.patch_size))
    train()
  else:
    stylize()


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
  eval_parser.add_argument('--patch-size', type=int, default=3,
                            help='size of patches for activations, default is 3')

  args = parser.parse_args()

  main()
