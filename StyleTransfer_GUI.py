import io
import os
import math
import sys

import wx
import cv2
from PIL import Image

import torch
from torch.autograd import Variable
from torch import nn

import torchvision.transforms as transforms
import torchvision.models as models

# run on GPU
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Mean and Standard deviation of the Imagenet dataset
MEAN_IMAGE = [0.485, 0.456, 0.406]
STD_IMAGE = [0.229, 0.224, 0.225]

# helpers
toTensor = transforms.ToTensor()
toPILImage = transforms.ToPILImage()

num_accept_chars = "1234567890."


class Dumoulin_W(wx.Frame):
    def __init__(self, parent, title):
      super(Dumoulin_W, self).__init__(parent, title=title)
      exit_id = wx.NewId()
      self.Bind(wx.EVT_MENU, self.onW, id=exit_id)
      self.SetAcceleratorTable(wx.AcceleratorTable([(wx.ACCEL_CTRL, ord('W'), exit_id)]))
      self.style_model = None
      self.content = None
      self.sizer = wx.BoxSizer(wx.VERTICAL)

      # Part for displaying style images and taking in weights
      self.input_panel = wx.Panel(self, style=wx.TAB_TRAVERSAL)
      self.input_panel.style_sizer = wx.FlexGridSizer(2, 5, 3, 0)
      imgs = sorted(os.listdir(os.getcwd() + '/images/style_images'))
      if '.DS_Store' in imgs:
        imgs.remove('.DS_Store')
      for img in imgs:
        image = wx.Image(os.getcwd() + '/images/style_images/' + img, wx.BITMAP_TYPE_ANY).Scale(200, 200)
        self.input_panel.style_sizer.Add(wx.StaticBitmap(self, bitmap=wx.Bitmap(image)), 1, wx.EXPAND)
      self.weights = []
      for i in range(len(imgs)):
        self.weights.append(wx.TextCtrl(self, value='0', style=wx.TE_CENTER))
        self.weights[i].Bind(wx.EVT_CHAR, self.onNum)
        self.weights[i].Bind(wx.EVT_SET_FOCUS, self.onFocus)
        self.weights[i].Bind(wx.EVT_KILL_FOCUS, self.lostFocus)
        self.input_panel.style_sizer.Add(self.weights[i], 1, wx.EXPAND)
      self.sizer.Add(self.input_panel.style_sizer, 1, wx.EXPAND)

      self.sizer.Add(wx.StaticText(self, label='Please specify the weights (positive numbers) of each style you want'
                                               ' to be seen in the result:'),
                     0, wx.ALIGN_CENTER)

      # Part for choosing content image
      dc = wx.MemoryDC(wx.Bitmap(256, 256))
      dc.SetFont(wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
      text = 'Click to choose content image'
      tw, th = dc.GetTextExtent(text)
      dc.DrawText(text, (256 - tw) / 2, (256 - th) / 2)
      dc.DrawLineList(((0, 0, 0, 255), (0, 0, 255, 0), (255, 255, 0, 255), (255, 255, 255, 0)))
      self.image_ctrl = wx.StaticBitmap(self, bitmap=dc.GetAsBitmap())
      self.image_ctrl.Bind(wx.EVT_LEFT_UP, self.onImage)
      self.sizer.Add(self.image_ctrl, 0, wx.CENTER)

      self.capture_button = wx.Button(self, label='Capture camera image', size=(256,40))
      self.capture_button.Bind(wx.EVT_BUTTON, self.onCapture)
      self.sizer.Add(self.capture_button, 0, wx.CENTER)

      # Part for displaying the result
      self.styleize_button = wx.Button(self, label='Stylize', size=(256, 40))
      self.styleize_button.Bind(wx.EVT_BUTTON, self.onStylize)
      self.sizer.Add(self.styleize_button, 0, wx.CENTER, 20)

      self.SetSizer(self.sizer)
      self.SetAutoLayout(1)
      self.sizer.Fit(self)

      self.Center()

    def onCapture(self, e):
      self.view = wx.Frame(self, title='Capture image')
      exit_id = wx.NewId()
      self.view.Bind(wx.EVT_MENU, self.onW, id=exit_id)
      self.view.SetAcceleratorTable(wx.AcceleratorTable([(wx.ACCEL_CTRL, ord('W'), exit_id),
                                                         (wx.ACCEL_CTRL, ord('Q'), exit_id)]))
      self.view.Bind(wx.EVT_CLOSE, self.onCloseCap)

      sizer = wx.BoxSizer(wx.VERTICAL)
      self.frame_ctrl = wx.StaticBitmap(self.view, bitmap=wx.Bitmap(800,500))
      sizer.Add(self.frame_ctrl, 1, wx.EXPAND)
      button = wx.Button(self.view, label='Capture')
      button.Bind(wx.EVT_BUTTON, self.onCap)
      sizer.Add(button, 0, wx.EXPAND)

      self.view.SetSizer(sizer)
      self.view.SetAutoLayout(1)
      sizer.Fit(self.view)

      self.stream = io.BytesIO()
      self.cap = cv2.VideoCapture(0)
      self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
      self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

      self.timer = wx.Timer(self.view)
      self.view.Bind(wx.EVT_TIMER, self.onNextImage, self.timer)
      self.timer.Start(30)

      self.view.Show()
      self.view.Center()

    def onNextImage(self, e):
      _, image = self.cap.read()
      self.stream = io.BytesIO()
      Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(self.stream, format='JPEG')
      self.frame_ctrl.SetBitmap(wx.Bitmap(wx.Image(io.BytesIO(self.stream.getvalue())).Scale(800, 500)))
    
    def onCap(self, e):
      self.image_ctrl.SetBitmap(wx.Bitmap(wx.Image(io.BytesIO(self.stream.getvalue())).Scale(256, 256)))
      self.content = self.stream.getvalue()
      e.GetEventObject().GetParent().Close()

    def onCloseCap(self, e):
      self.timer.Destroy()
      self.cap.release()
      e.Skip()

    def onFocus(self, e):
      if e.GetEventObject().GetValue() == '0':
        e.GetEventObject().SetValue('')
      e.Skip()

    def lostFocus(self, e):
      if e.GetEventObject().GetValue() == '':
        e.GetEventObject().SetValue('0')
      e.Skip()

    def onStylize(self, e):
      if self.content is None:
        wx.MessageBox('Please choose the image to be transferred', 'Sorry', parent=self)
        return False

      weights = [float(x.GetValue() if (x.GetValue() != '') else '0') for x in self.weights]
      if max(weights) == 0 or min(weights) < 0:
        wx.MessageBox('Please ensure at least one of the weight is positive, and none of them is negative',
                      'Sorry', parent=self)
        return False

      view = wx.Frame(self, title='Stylized image')
      exit_id = wx.NewId()
      view.Bind(wx.EVT_MENU, self.onW, id=exit_id)
      view.SetAcceleratorTable(wx.AcceleratorTable([(wx.ACCEL_CTRL, ord('W'), exit_id)]))

      sizer = wx.BoxSizer(wx.VERTICAL)

      # where stylization happens
      sum = math.fsum(weights)
      weights = [x / sum for x in weights]
      self.result = self.stylize(weights)

      sizer.Add(wx.StaticBitmap(view, bitmap=wx.Bitmap(wx.Image(io.BytesIO(self.result)))), 1, wx.EXPAND)
      save_button = wx.Button(view, label='Save result')
      save_button.Bind(wx.EVT_BUTTON, self.onSave)
      sizer.Add(save_button, 0, wx.EXPAND)

      view.SetSizer(sizer)
      view.SetAutoLayout(1)
      sizer.Fit(view)

      view.Center()
      view.Show()

    def onW(self, e):
      e.GetEventObject().Close()

    def onSave(self, e):
      dirname = os.getcwd() + '/images/'
      dlg = wx.FileDialog(self, defaultDir=dirname, wildcard="*.*", style=wx.FD_SAVE)
      if dlg.ShowModal() == wx.ID_OK:
        filename = dlg.GetFilename()
        if '.jpg' not in filename:
          filename += '.jpg'
        dirname = dlg.GetDirectory()
        with open(os.path.join(dirname, filename), 'wb') as file:
          file.write(self.result)
      dlg.Destroy()

    def onImage(self, e):
      dirname = os.getcwd() + '/images/content_images'
      dlg = wx.FileDialog(self, "Choose an image", dirname, "", "*.*", wx.FD_OPEN)
      if dlg.ShowModal() == wx.ID_OK:
        filename = dlg.GetFilename()
        dirname = dlg.GetDirectory()
        with open(os.path.join(dirname, filename), 'rb') as file:
          self.content = file.read()
        self.image_ctrl.SetBitmap(wx.Bitmap(wx.Image(os.path.join(dirname, filename)).Scale(256, 256)))
      dlg.Destroy()

    def onNum(self, e):
      c = e.GetKeyCode()
      text = e.GetEventObject().GetValue()
      if c == 13:
        self.onStylize(None)
      if len(text) < 11 and chr(c) in num_accept_chars and (chr(c) != '.' or '.' not in text):
        e.Skip()
      else:
        return False

    def stylize(self, weights):
      content_image = self.image_loader()
      content_image = content_image.unsqueeze(0)
      content_image = Variable(content_image, volatile=True)

      if self.style_model is None:
        self.style_model = TransformerNetwork(len(weights))
        self.style_model.load_state_dict(torch.load(os.getcwd() +
                                                    '/model/dance_mandolin_girl_scream_starry_night_wave.model'))
        if use_cuda:
          self.style_model.cuda()

      output = self.style_model(content_image, training=False, weights_of_styles=weights)
      image = io.BytesIO()
      toPILImage(output.data.squeeze()).save(image, format='JPEG')
      return image.getvalue()

    def image_loader(self, size=None, transform=None):
      """ helper for loading images
      Args:
        size: the size of the loaded image needed to be resized to, can be a sequence or int
        transform: the transform that needed to be done on the image
      Returns:
        image: the Tensor representing the image loaded (value in [0,1])
      """
      image = Image.open(io.BytesIO(self.content))
      if transform is not None:
        image = toTensor(transform(image))
      elif size is not None:
        cutImage = transforms.Resize(size)
        image = toTensor(cutImage(image))
      else:
        image = toTensor(image)
      return image.type(dtype)

class Chen_W(wx.Frame):
  def __init__(self, parent, title):
    super(Chen_W, self).__init__(parent, title=title)
    exit_id = wx.NewId()
    self.Bind(wx.EVT_MENU, self.onW, id=exit_id)
    self.SetAcceleratorTable(wx.AcceleratorTable([(wx.ACCEL_CTRL, ord('W'), exit_id)]))
    self.sizer = wx.BoxSizer(wx.VERTICAL)

    self.inverse_net = None
    self.vgg = None
    self.content = None
    self.style = None

    self.image_sizer = wx.BoxSizer(wx.HORIZONTAL)
    # Content image
    dc = wx.MemoryDC(wx.Bitmap(500, 500))
    dc.SetFont(wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
    text = 'Click to choose content image'
    tw, th = dc.GetTextExtent(text)
    dc.DrawText(text, (500 - tw) / 2, (500 - th) / 2)
    dc.DrawLineList(((0, 0, 0, 499), (0, 0, 499, 0), (499, 499, 0, 499), (499, 499, 499, 0)))
    bm = wx.StaticBitmap(self, bitmap=dc.GetAsBitmap())
    bm.Bind(wx.EVT_LEFT_UP, self.onContentImage)
    self.image_sizer.Add(bm, 0, wx.EXPAND)
    # Style image
    dc2 = wx.MemoryDC(wx.Bitmap(500, 500))
    dc2.SetFont(wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
    text = 'Click to choose style image'
    tw, th = dc2.GetTextExtent(text)
    dc2.DrawText(text, (500 - tw) / 2, (500 - th) / 2)
    dc2.DrawLineList(((0, 0, 0, 499), (0, 0, 499, 0), (499, 499, 0, 499), (499, 499, 499, 0)))
    bm2 = wx.StaticBitmap(self, bitmap=dc2.GetAsBitmap())
    bm2.Bind(wx.EVT_LEFT_UP, self.onStyleImage)
    self.image_sizer.Add(bm2, 0, wx.EXPAND)

    self.sizer.Add(self.image_sizer, 0, wx.EXPAND)

    button = wx.Button(self, label='Stylize', size=((500 - 5) * 2, 40))
    button.Bind(wx.EVT_BUTTON, self.onStylize)
    self.sizer.Add(button, 0, wx.CENTER)

    self.SetSizer(self.sizer)
    self.SetAutoLayout(1)
    self.sizer.Fit(self)

    self.Center()

  def onStylize(self, e):
    if self.content is None or self.style is None:
      wx.MessageBox('Please choose the images to be transferred', 'Sorry', parent=self)
      return False

    view = wx.Frame(self, title='Stylized image')
    exit_id = wx.NewId()
    view.Bind(wx.EVT_MENU, self.onW, id=exit_id)
    view.SetAcceleratorTable(wx.AcceleratorTable([(wx.ACCEL_CTRL, ord('W'), exit_id)]))

    sizer = wx.BoxSizer(wx.VERTICAL)

    self.result = self.stylize()

    sizer.Add(wx.StaticBitmap(view, bitmap=wx.Bitmap(wx.Image(io.BytesIO(self.result)))), 1, wx.EXPAND)
    save_button = wx.Button(view, label='Save result')
    save_button.Bind(wx.EVT_BUTTON, self.onSave)
    sizer.Add(save_button, 0, wx.EXPAND)

    view.SetSizer(sizer)
    view.SetAutoLayout(1)
    sizer.Fit(view)

    view.Center()
    view.Show()

  def stylize(self):
    content_image = Variable(self.image_loader(self.content).unsqueeze_(0).type(dtype), volatile=True)
    style_image = Variable(self.image_loader(self.style).unsqueeze_(0).type(dtype), volatile=True)

    if self.inverse_net is None:
      self.inverse_net = InverseNet()
      self.inverse_net.load_state_dict(torch.load(os.getcwd() + '/model/inverse_net.model'))
      if use_cuda:
        self.inverse_net.cuda()
    if self.vgg is None:
      self.vgg = VGG()
      if use_cuda:
        self.vgg.cuda()

    content_activation = self.vgg(self.normalize_images(content_image))
    style_activation = self.vgg(self.normalize_images(style_image))
    target_activation = Variable(self.style_swap(content_activation, style_activation), volatile=True)

    output = self.inverse_net(target_activation)  # target_activation
    image = io.BytesIO()
    toPILImage(self.denormalize_image(output.data.squeeze())).save(image, format='JPEG')
    return image.getvalue()

  def onContentImage(self, e):
    dirname = os.getcwd() + '/images/content_images'
    dlg = wx.FileDialog(self, "Choose an image", dirname, "", "*.*", wx.FD_OPEN)
    if dlg.ShowModal() == wx.ID_OK:
      filename = dlg.GetFilename()
      dirname = dlg.GetDirectory()
      with open(os.path.join(dirname, filename), 'rb') as file:
        self.content = file.read()
      e.GetEventObject().SetBitmap(wx.Bitmap(wx.Image(os.path.join(dirname, filename)).Scale(500, 500)))
    dlg.Destroy()

  def onStyleImage(self, e):
    dirname = os.getcwd() + '/images/style_images'
    dlg = wx.FileDialog(self, "Choose an image", dirname, "", "*.*", wx.FD_OPEN)
    if dlg.ShowModal() == wx.ID_OK:
      filename = dlg.GetFilename()
      dirname = dlg.GetDirectory()
      with open(os.path.join(dirname, filename), 'rb') as file:
        self.style = file.read()
      e.GetEventObject().SetBitmap(wx.Bitmap(wx.Image(os.path.join(dirname, filename)).Scale(500, 500)))
    dlg.Destroy()

  def onSave(self, e):
    dirname = os.getcwd() + '/images/'
    dlg = wx.FileDialog(self, defaultDir=dirname, wildcard="*.*", style=wx.FD_SAVE)
    if dlg.ShowModal() == wx.ID_OK:
      filename = dlg.GetFilename()
      if '.jpg' not in filename:
        filename += '.jpg'
      dirname = dlg.GetDirectory()
      with open(os.path.join(dirname, filename), 'wb') as file:
        file.write(self.result)
    dlg.Destroy()

  def image_loader(self, image, size=None, transform=None):
    """ helper for loading images
    Args:
      size: the size of the loaded image needed to be resized to, can be a sequence or int
      transform: the transform that needed to be done on the image
    Returns:
      image: the Tensor representing the image loaded (value in [0,1])
    """
    image = Image.open(io.BytesIO(image))
    if transform is not None:
      image = toTensor(transform(image))
    elif size is not None:
      cutImage = transforms.Resize(size)
      image = toTensor(cutImage(image))
    else:
      image = toTensor(image)
    return image.type(dtype)

  def style_swap(content_activation, style_activation):
    patch_size = 3
    _, ch_c, h_c, w_c = content_activation.size()
    _, ch_s, h_s, w_s = style_activation.size()
    if ch_c != ch_s:
      print("ERROR: the layer for content activation and style activation should be the same", flush=True)
      sys.exit(1)

    # Convolutional layer for performing the calculation of cosine measures
    conv = nn.Conv3d(1, (h_s - int(patch_size / 2) * 2) * (w_s - int(patch_size / 2) * 2),
                     kernel_size=(ch_s, patch_size, patch_size))
    for param in conv.parameters():
      param.requires_grad = False
    if use_cuda:
      conv.cuda()

    for h in range(h_s - int(patch_size / 2) * 2):
      for w in range(w_s - int(patch_size / 2) * 2):
        conv.weight.data[h * (w_s - int(patch_size / 2) * 2) + w, 0, :, :, :] = nn.functional.normalize(
          style_activation.data[:, :, h:h + patch_size, w:w + patch_size].squeeze())
    conv.bias.data.mul_(0)

    # Convolution and taking the maximum of cosine mearsures
    k = conv(content_activation.unsqueeze(0))
    _, max_index = k.squeeze().max(0)

    # Constructing target activation
    overlaps = torch.zeros(h_c, w_c).type(dtype)
    target_activation = torch.zeros(content_activation.size()).type(dtype)
    for h in range(h_c - int(patch_size / 2) * 2):
      for w in range(w_c - int(patch_size / 2) * 2):
        s_w = int(max_index.data[h, w] % (w_s - int(patch_size / 2) * 2))
        s_h = int(max_index.data[h, w] // (w_s - int(patch_size / 2) * 2))
        target_activation[:, :, h:h + patch_size, w:w + patch_size] = \
          target_activation[:, :, h:h + patch_size, w:w + patch_size] + \
          style_activation.data[:, :, s_h:s_h + patch_size, s_w:s_w + patch_size]
        overlaps[h:h + patch_size, w:w + patch_size].add_(1)

    return target_activation.div(overlaps)

  def normalize_images(self, images):
    """ Normalised a batch of images wrt the Imagenet dataset """
    # normalize using imagenet mean and std
    mean = torch.zeros(images.data.size()).type(dtype)
    std = torch.zeros(images.data.size()).type(dtype)
    for i in range(3):
      mean[:, i, :, :] = MEAN_IMAGE[i]
      std[:, i, :, :] = STD_IMAGE[i]
    return (images - Variable(mean, requires_grad=False)) / Variable(std, requires_grad=False)

  def denormalize_image(self, image):
    """ Denormalised the image wrt the Imagenet dataset """
    # denormalize using imagenet mean and std
    mean = torch.zeros(image.size()).type(dtype)
    std = torch.zeros(image.size()).type(dtype)
    for i in range(3):
      mean[i, :, :] = MEAN_IMAGE[i]
      std[i, :, :] = STD_IMAGE[i]
    return (image * std) + mean

  def onW(self, e):
    e.GetEventObject().Close()

class MainWindow(wx.Frame):
  def __init__(self, parent, title):
    super(MainWindow, self).__init__(parent, title=title)
    self.sizer = wx.BoxSizer(wx.HORIZONTAL)

    # Use trained styles
    dc = wx.MemoryDC(wx.Bitmap(360, 360))
    image = wx.Image(os.getcwd() + '/images/style_images/mandolin_girl.jpg', wx.BITMAP_TYPE_ANY).Scale(360, 360).Blur(4)
    dc.DrawBitmap(wx.Bitmap(image), 0, 0)

    dc.SetFont(wx.Font(28, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
    text = 'Use trained styles'
    tw, th = dc.GetTextExtent(text)
    dc.DrawText(text, (360 - tw) / 2, (360 - th) / 2)

    bm = wx.StaticBitmap(self, bitmap=dc.GetAsBitmap())
    bm.Bind(wx.EVT_LEFT_UP, self.onTrained)
    self.sizer.Add(bm, 1, wx.EXPAND)

    # User specify style
    dc2 = wx.MemoryDC(wx.Bitmap(360, 360))
    dc2.SetFont(wx.Font(28, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
    text = 'Use your own style'
    tw, th = dc2.GetTextExtent(text)
    dc2.DrawText(text, (360 - tw) / 2, (360 - th) / 2)
    dc2.DrawLineList(((0, 0, 0, 359), (0, 0, 359, 0), (359, 359, 0, 359), (359, 359, 359, 0)))
    bm2 = wx.StaticBitmap(self, bitmap=dc2.GetAsBitmap())
    bm2.Bind(wx.EVT_LEFT_UP, self.onCustomising)
    self.sizer.Add(bm2, 1, wx.EXPAND)

    self.SetSizer(self.sizer)
    self.SetAutoLayout(1)
    self.sizer.Fit(self)

    self.makeMenuBar()
    self.Center()

  def onTrained(self, e):
    Dumoulin_W(self, 'Trained styles').Show()

  def onCustomising(self, e):
    Chen_W(self, 'Own style').Show()

  def makeMenuBar(self):
    # Setting up the menu
    filemenu = wx.Menu()
    menuAbout = filemenu.Append(wx.ID_ABOUT, "&About Style Transfer", " Information about this program")
    filemenu.AppendSeparator()
    menuExit = filemenu.Append(wx.ID_EXIT, "E&xit", " Terminate the program")

    # Creating the menubar
    menuBar = wx.MenuBar()
    menuBar.Append(filemenu, "&Style Transfer")
    self.SetMenuBar(menuBar)

    # Set events.
    self.Bind(wx.EVT_MENU, self.onAbout, menuAbout)
    self.Bind(wx.EVT_MENU, self.onExit, menuExit)

  def onAbout(self, e):
    dlg = wx.MessageDialog(self, "App for doing style transfers", "About Style Transfer")
    dlg.ShowModal()
    dlg.Destroy()

  def onExit(self, e):
    self.Close()

class TransformerNetwork(nn.Module):
  ''' Module implementing the image transformation network '''
  def __init__(self, styles):
    super(TransformerNetwork, self).__init__()
    # Downsampling
    self.conv1 = nn.Conv2d(3, 32, kernel_size=9)
    self.in1 = InstanceNorm(32, styles)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
    self.in2 = InstanceNorm(64, styles)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
    self.in3 = InstanceNorm(128, styles)
    # Residual layers
    self.res1 = ResidualBlock(128, styles)
    self.res2 = ResidualBlock(128, styles)
    self.res3 = ResidualBlock(128, styles)
    self.res4 = ResidualBlock(128, styles)
    self.res5 = ResidualBlock(128, styles)
    # Upsampling
    self.deconv1 = nn.Conv2d(128, 64, kernel_size=3)
    self.in4 = InstanceNorm(64, styles)
    self.deconv2 = nn.Conv2d(64, 32, kernel_size=3)
    self.in5 = InstanceNorm(32, styles)
    self.deconv3 = nn.Conv2d(32, 3, kernel_size=9)
    self.in6 = InstanceNorm(3, styles)
    # Non-linearities
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    # Reflection paddings
    self.pad_4 = nn.ReflectionPad2d(4)
    self.pad_1 = nn.ReflectionPad2d(1)
    # Upsample
    self.up = nn.Upsample(scale_factor=2)

  def forward(self, x, training, style=None, weights_of_styles=None):
    y = self.relu(self.in1(self.conv1(self.pad_4(x)), training, style, weights_of_styles))
    # _, _, h_size1, w_size1 = y.size()
    y = self.relu(self.in2(self.conv2(self.pad_1(y)), training, style, weights_of_styles))
    # _, _, h_size2, w_size2 = y.size()
    y = self.relu(self.in3(self.conv3(self.pad_1(y)), training, style, weights_of_styles))
    y = self.res1(y, training, style, weights_of_styles)
    y = self.res2(y, training, style, weights_of_styles)
    y = self.res3(y, training, style, weights_of_styles)
    y = self.res4(y, training, style, weights_of_styles)
    y = self.res5(y, training, style, weights_of_styles)
    # upsample1 = nn.Upsample(size=(h_size2, w_size2))
    y = self.relu(self.in4(self.deconv1(self.pad_1(self.up(y))), training, style, weights_of_styles))
    # upsample2 = nn.Upsample(size=(h_size1, w_size1))
    y = self.relu(self.in5(self.deconv2(self.pad_1(self.up(y))), training, style, weights_of_styles))
    y = self.sigmoid(self.in6(self.deconv3(self.pad_4(y)), training, style, weights_of_styles))
    return y

class ResidualBlock(nn.Module):
  ''' Residual block
      It helps the network to learn the identify function
  '''
  def __init__(self, channels, styles):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1)
    self.in1 = InstanceNorm(channels, styles)
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1)
    self.in2 = InstanceNorm(channels, styles)
    self.relu = nn.ReLU()
    self.pad_1 = nn.ReflectionPad2d(1)

  def forward(self, x, training, style=None, weights_of_style=None):
    y = self.relu(self.in1(self.conv1(self.pad_1(x)), training, style, weights_of_style))
    y = self.in2(self.conv2(self.pad_1(y)), training, style, weights_of_style)
    return y + x

class InstanceNorm(nn.Module):
  ''' Module encapsulates n affine instance normalisations '''
  def __init__(self, channels, styles):
    super(InstanceNorm, self).__init__()
    self.channels = channels
    self.styles = styles
    self.norms = nn.ModuleList([nn.InstanceNorm2d(channels, affine=True) for i in range(styles)])

  def forward(self, x, training, style=None, weights_of_styles=None):
    if training:
      if style is None:
        raise RuntimeError('specify which style to apply when training')
      return self.norms[style](x)

    if weights_of_styles is None or len(weights_of_styles) != self.styles:
      raise RuntimeError('specify the weight for each style, in order')
    layer = nn.InstanceNorm2d(self.channels, affine=True)
    layer.weight.data = torch.zeros(self.channels)
    layer.bias.data = torch.zeros(self.channels)
    for i in range(self.styles):
      layer.weight.data += self.norms[i].weight.data * weights_of_styles[i]
      layer.bias.data += self.norms[i].bias.data * weights_of_styles[i]
    return layer(x)

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

if __name__ == '__main__':
  app = wx.App()
  frame = MainWindow(None, "Style Transfer")
  frame.Show()
  app.MainLoop()
