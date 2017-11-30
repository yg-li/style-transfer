import os
import argparse
import copy
import logging

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as utils

from PIL import Image

# how important is the content of content image and that of the generated image being similar
CONTENT_WEIGHT = 50
# how important is the style of style image and that of the generated image being similar
STYLE_WEIGHT = 3750
TV_WEIGHT = 1e-6

NUM_STEPS = 500
SHOW_STEPS = 20

# run on GPU
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Mean and Standard deviation of the Imagenet dataset
MEAN_IMAGE = [0.485, 0.456, 0.406]
# helpers
toTensor = transforms.ToTensor()
normaliseImage = transforms.Normalize(MEAN_IMAGE, [1,1,1])
deNormaliseImage = transforms.Normalize([-x for x in MEAN_IMAGE], [1,1,1])

def image_loader(image_name, height=None, width=None):
  """ helper for loading images
  Args:
    image_name: the path to the image
    height, width: the height and width the loaded image needed to be resized to
  Returns:
    image: the Variable representing the image loaded (value in [0,1])
  """
  image = Image.open(image_name)
  if height is None or width is None:
    image = Variable(normaliseImage(toTensor(image)))
  else:
    image = Variable(normaliseImage(toTensor(image.resize((width, height)))))
  image = image.unsqueeze(0) # make the tensor to be 4D so that VGG can process it
  return image.type(dtype)

class LossNet(nn.Module):
  def __init__(self):
    super(LossNet, self).__init__()
    vgg = models.vgg19(pretrained=True).features
    self.slice1 = nn.Sequential()
    self.slice2 = nn.Sequential()
    self.slice3 = nn.Sequential()
    self.slice4 = nn.Sequential()
    self.slice5 = nn.Sequential()
    self.slice6 = nn.Sequential()
    for x in range(2):
      self.slice1.add_module(str(x), vgg[x])
    for x in range(2, 7):
      if x == 4:
        self.slice2.add_module(str(x), nn.AvgPool2d(kernel_size=vgg[x].kernel_size, stride=vgg[x].stride,
                                                    padding = vgg[x].padding))
      else:
        self.slice2.add_module(str(x), vgg[x])
    for x in range(7, 12):
      if x == 9:
        self.slice3.add_module(str(x), nn.AvgPool2d(kernel_size=vgg[x].kernel_size, stride=vgg[x].stride,
                                                    padding = vgg[x].padding))
      else:
        self.slice3.add_module(str(x), vgg[x])
    for x in range(12, 21):
      if x == 18:
        self.slice4.add_module(str(x), nn.AvgPool2d(kernel_size=vgg[x].kernel_size, stride=vgg[x].stride,
                                                    padding = vgg[x].padding))
      else:
        self.slice4.add_module(str(x), vgg[x])
    for x in range(21, 23):
      self.slice5.add_module(str(x), vgg[x])
    for x in range(23, 30):
      if x == 27:
        self.slice6.add_module(str(x), nn.AvgPool2d(kernel_size=vgg[x].kernel_size, stride=vgg[x].stride,
                                                    padding = vgg[x].padding))
      else:
        self.slice6.add_module(str(x), vgg[x])
    for param in self.parameters():
      param.requires_grad = False

  def forward(self, x):
    relu1_1 = self.slice1(x)
    relu2_1 = self.slice2(relu1_1)
    relu3_1 = self.slice3(relu2_1)
    relu4_1 = self.slice4(relu3_1)
    relu4_2 = self.slice5(relu4_1)
    relu5_1 = self.slice6(relu4_2)
    return relu4_2, [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]

def gram_matrix(input):
  """ Compute batch-wise gram matrices """
  b, ch, h, w = input.size()
  features = input.view(b, ch, h * w)
  G = features.bmm(features.transpose(1,2))
  return G / (h * w)

def transfer(content_img, style_img, input_img, output_dir, num_steps=NUM_STEPS):
  """ the main method for doing the style transfer
  Args:
    vgg: the feature component of a pre-trained VGG-19 network
    content_img: the content image
    style_img: the style image
    input_img: the initial generated image
    output_dir: the path to the directory storing the output of style transfer
    num_steps: the maximum number of iterations for updating the generated image
  """
  print('Building the style transfer model...', flush=True)
  lossnet = LossNet()
  if use_cuda:
    lossnet = lossnet.cuda()
  target_feature_content, _ = lossnet(content_img)
  _, target_feature_style = lossnet(style_img)
  target_gram_style = [gram_matrix(x) for x in target_feature_style]

  input_param = nn.Parameter(input_img.data)
  optimizer = optim.LBFGS([input_param])
  mse = nn.MSELoss()

  print('Transfering style...', flush=True)
  run = [NUM_STEPS-num_steps]
  while run[0] < NUM_STEPS:
    def closure():
      # Clamp the image to be in ([0,1] - MEAN_IMAGE)
      for i in range(3):
        input_param.data[0][i].clamp_(0-MEAN_IMAGE[i], 1-MEAN_IMAGE[i])
      if run[0] % SHOW_STEPS == 0:
        utils.save_image(deNormaliseImage(copy.deepcopy(input_param.data[0])).clamp(0, 1), output_dir + str(run[0]) + '.jpg')

      optimizer.zero_grad()
      feature_content, feature_style = lossnet(input_param)
      style_loss = 0.

      content_loss = CONTENT_WEIGHT * mse(feature_content, target_feature_content)

      gram_style = [gram_matrix(x) for x in feature_style]
      for gm_s, tgm_s in zip(gram_style, target_gram_style):
        style_loss += STYLE_WEIGHT * mse(gm_s, tgm_s)

      tv_loss = TV_WEIGHT * (torch.sum(torch.abs(input_param[:,:,:,:-1]-input_param[:,:,:,1:])) +
                              torch.sum(torch.abs(input_param[:,:,:-1,:]-input_param[:,:,1:,:])))

      total_loss = content_loss + style_loss + tv_loss
      total_loss.backward()

      run[0] += 1
      if run[0] % SHOW_STEPS == 0:
        mesg = 'run {}:\nStyle Loss : {:4f} Content Loss: {:4f}'.format(run[0], style_loss.data[0], content_loss.data[0])
        logging.info(mesg)
        print(mesg, flush=True)

      return total_loss

    optimizer.step(closure)

  return deNormaliseImage(input_param.data[0]).clamp(0, 1)

def main(args):
  num_steps = NUM_STEPS

  global STYLE_WEIGHT

  # set content and style weights
  if args.style_to_content is not None:
    STYLE_WEIGHT = CONTENT_WEIGHT * args.style_to_content

  # load images
  content_img = image_loader(os.getcwd() + '/images/content_images/' + args.content_image)
  _, _, img_height, img_width = content_img.size()
  style_img = image_loader(os.getcwd() + '/images/style_images/' + args.style_image, img_height, img_width)
  assert content_img.size() == style_img.size(), 'the content and style images should have the same size'

  # start transferring from content image
  input_img = Variable(torch.randn(content_img.size())).type(dtype) # content_img.clone()

  # name of dir of output is (name_of_content_img + name_of_style_img)
  output_dir = os.getcwd() + '/images/output_images/' + \
               args.content_image.split('.')[0] + '_' + args.style_image.split('.')[0] + '/'
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  elif 'result.jpg' in os.listdir(output_dir):
    print('Already transferred\n', flush=True)
    return
  else: # start transferring from an intermediate stage
    files = os.listdir(output_dir)
    for i in reversed(range(int(num_steps/SHOW_STEPS))):
      if str(i*SHOW_STEPS)+'.jpg' in files:
        try:
          input_img = image_loader(output_dir+str(i*SHOW_STEPS)+'.jpg', img_height, img_width)
          num_steps -= i*SHOW_STEPS
          print('Starting transferring from the', str(i*SHOW_STEPS), 'iteration', flush=True)
          break
        except IOError:
          continue

  # start logging
  logging.basicConfig(filename=output_dir + 'log', level=logging.INFO)
  logging.info('Style weight: ' + str(STYLE_WEIGHT) + ' Content Weight: ' + str(CONTENT_WEIGHT))

  # transfer style
  output_img = transfer(content_img, style_img, input_img, output_dir, num_steps)
  utils.save_image(output_img, output_dir + 'result.jpg')
  print('Done with transferring\n', flush=True)

  logging.info("Transferring finished")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('content_image', help='name of content image', type=str)
  parser.add_argument('style_image', help='name of style image', type=str)
  parser.add_argument('--style-to-content', help='the ratio of weights for style and content similarities', type=int)

  args = parser.parse_args()
  main(args)