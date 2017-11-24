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

# layers we used to represent content
CONTENT_LAYERS = ['relu4_2']
# layers we used to represent style
STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
# how important is the content of content image and that of the generated image being similar
CONTENT_WEIGHT = 1
# how important is the style of style image and that of the generated image being similar
STYLE_WEIGHT = 1000
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
  return image

class ContentLoss(nn.Module):
  """ the module helps to compute the content loss """
  def __init__(self, target, weight):
    super(ContentLoss,self).__init__()
    self.target = target.detach() * weight
    self.weight = weight
    self.criterion = nn.MSELoss()

  def forward(self, input):
    self.loss = self.criterion(input * self.weight, self.target)
    self.output = input
    return self.output

  def backward(self, retain_graph=True):
    self.loss.backward(retain_graph=retain_graph)
    return self.loss

class GramMatrix(nn.Module):
  """ module for computing gram matrix of a layer of activations """
  def forward(self, input):
    b, c, h, w = input.size()

    features = input.view(b * c, h * w)

    G = torch.mm(features, features.t())  # compute the gram matrix

    return G.div(b * c * h * w)

class StyleLoss(nn.Module):
  """ the module helps to compute the style loss at a layer """
  def __init__(self, target, weight):
    super(StyleLoss,self).__init__()
    self.target = target.detach() * weight
    self.weight = weight
    self.gram = GramMatrix()
    self.criterion = nn.MSELoss()

  def forward(self, input):
    self.output = input.clone()
    self.G = self.gram(input)
    self.G.mul_(self.weight)
    self.loss = self.criterion(self.G, self.target)
    return self.output

  def backward(self, retain_graph=True):
    self.loss.backward(retain_graph=retain_graph)
    return self.loss

def get_style_model_and_losses(vgg, content_img, style_img):
  """ constructs the model for style transfer as well as the losses
    Args:
      vgg: the feature component of a pre-trained VGG-19 network
      content_img: the content image
      style_img: the style image
    Returns:
      model: the style model
      content_losses: the list of modules for computing content loss
      style_losses: the list of modules for computing style loss
  """
  vgg = copy.deepcopy(vgg)

  content_losses = []
  style_losses = []

  model = nn.Sequential()  # the new model
  gram = GramMatrix()  # for computing the style targets

  if use_cuda:
    model = model.cuda()
    gram = gram.cuda()

  i = j = 1
  for layer in list(vgg):
    if isinstance(layer, nn.Conv2d):
      name = 'conv' + str(i) + '_' + str(j)
      model.add_module(name, layer)

    if isinstance(layer, nn.ReLU):
      name = 'relu' + str(i) + '_' + str(j)
      model.add_module(name, layer)

      if name in CONTENT_LAYERS:
        # add content loss:
        target = model(content_img).clone()
        content_loss = ContentLoss(target, CONTENT_WEIGHT)
        model.add_module('content_loss_' + str(i) + '_' + str(j), content_loss)
        content_losses.append(content_loss)

      if name in STYLE_LAYERS:
        # add style loss:
        target = model(style_img).clone()
        target_gram = gram(target)
        style_loss = StyleLoss(target_gram, STYLE_WEIGHT)
        model.add_module('style_loss_' + str(i) + '_' + str(j), style_loss)
        style_losses.append(style_loss)

      j += 1

    if isinstance(layer, nn.MaxPool2d):
      name = 'avg_pool_' + str(i)
      avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size,
                              stride=layer.stride, padding = layer.padding)
      model.add_module(name,avgpool)
      i += 1
      j = 1

  return model, content_losses, style_losses

def get_input_param_and_optimizer(input_img, max_iter=300):
  """ produce the input parameter (the generated image) and the optimizer
  Args:
    input_img: the initial generated image
    max_iter: the maximum number of iterations
  Returns:
    input_param: a parameter has the data of input_img
    optimizer: an L-BFGS optimizer
  """
  # input_img is a parameter that needs to be updated
  input_param = nn.Parameter(input_img.data)
  optimizer = optim.LBFGS([input_param], max_iter=max_iter)
  return input_param, optimizer

def run_style_transfer(vgg, content_img, style_img, input_img, output_dir, num_steps=300):
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
  model, content_losses, style_losses = get_style_model_and_losses(vgg, content_img, style_img)
  input_param, optimizer = get_input_param_and_optimizer(input_img, num_steps)

  print('Transfering style...', flush=True)
  run = [NUM_STEPS-num_steps]
  def closure():
    for i in range(3):
      input_param.data[0][i].clamp_(0-MEAN_IMAGE[i], 1-MEAN_IMAGE[i])
    if run[0] % SHOW_STEPS == 0:
      utils.save_image(deNormaliseImage(copy.deepcopy(input_param.data[0])).clamp(0, 1), output_dir + str(run[0]) + '.jpg')

    optimizer.zero_grad()
    model(input_param)
    style_score = 0
    content_score = 0

    for cl in content_losses:
      content_score += cl.backward()
    for sl in style_losses:
      style_score += sl.backward()

    tv_score = TV_WEIGHT * (torch.sum(torch.abs(input_param[:,:,:,:-1]-input_param[:,:,:,1:])) +
                            torch.sum(torch.abs(input_param[:,:,:-1,:]-input_param[:,:,1:,:])))

    run[0] += 1
    if run[0] % SHOW_STEPS == 0:
      logging.info("run {}:".format(run))
      logging.info('Style Loss : {:4f} Content Loss: {:4f}'.format(
        style_score.data[0], content_score.data[0]))


    return content_score + style_score + tv_score

  optimizer.step(closure)

  return deNormaliseImage(input_param.data[0]).clamp(0, 1)

def main(args):
  num_steps = NUM_STEPS

  # set content and style weights
  if args.style_to_content is not None:
    STYLE_WEIGHT = CONTENT_WEIGHT * args.style_to_content

  # load images
  content_img = image_loader(os.getcwd() + '/images/content_images/' + args.content_image).type(dtype)
  _, _, img_height, img_width = content_img.size()
  style_img = image_loader(os.getcwd() + '/images/style_images/' + args.style_image, img_height, img_width).type(dtype)
  assert content_img.size() == style_img.size(), 'the content and style images should have the same size'

  # start transferring from content image
  input_img = content_img.clone()  # Variable(torch.rand(content_img.size())).type(dtype)

  # name of dir of output is (name_of_content_img + name_of_style_img)
  output_dir = os.getcwd() + '/images/output_images/' + \
               args.content_image.split('.')[0] + '_' + args.style_image.split('.')[0] + '/'
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  elif 'result.jpg' in os.listdir(output_dir):
    print('Already transferred', flush=True)
    return
  else: # start transferring from an intermediate stage
    files = os.listdir(output_dir)
    for i in reversed(range(num_steps/SHOW_STEPS)):
      if str(i*SHOW_STEPS)+'.jpg' in files:
        try:
          input_img = image_loader(output_dir+str(i*SHOW_STEPS)+'.jpg', img_height, img_width).type(dtype)
          num_steps -= i*SHOW_STEPS
          print('Starting transferring from the', str(i*SHOW_STEPS), 'iteration', flush=True)
          break
        except IOError:
          continue

  vgg = models.vgg19(pretrained=True).features
  if use_cuda:
    vgg = vgg.cuda()

  # start logging
  logging.basicConfig(filename=output_dir + 'log', level=logging.DEBUG)

  # transfer style
  output_img = run_style_transfer(vgg, content_img, style_img, input_img, output_dir, num_steps)
  utils.save_image(output_img, output_dir + 'result.jpg')
  print('Done with transferring\n', flush=True)

  logging.info("Transferring finished")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('content_image', help='name of content image', type=str)
  parser.add_argument('style_image', help='name of style image', type=str)
  parser.add_argument('--style-to-content', help='the ratio of weights for style and content similarity', type=int)

  args = parser.parse_args()
  main(args)
