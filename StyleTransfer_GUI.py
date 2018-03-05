import io
import os
import math

import wx
from PIL import Image

import torch
from torch.autograd import Variable
from torch import nn

import torchvision.transforms as transforms

# run on GPU
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# helpers
toTensor = transforms.ToTensor()
toPILImage = transforms.ToPILImage()

num_accept_chars = "1234567890."


class MainWindow(wx.Frame):
    def __init__(self, parent, title):
      super(MainWindow, self).__init__(parent, title=title)
      self.style_model = None
      self.content = None
      self.sizer = wx.BoxSizer(wx.VERTICAL)

      # Part for displaying style images and taking in weights
      self.sizer.Add(wx.StaticText(self, label='Please specify the weights (positive numbers) of each style you want'
                                               ' to be seen in the result:'),
                     0, wx.ALIGN_CENTER)
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

      # Part for choosing content image
      dc = wx.MemoryDC(wx.Bitmap(256, 256))
      text = 'Click here to choose the input image'
      tw, th = dc.GetTextExtent(text)
      dc.DrawText(text, (256 - tw) / 2, (256 - th) / 2)
      self.image_ctrl = wx.StaticBitmap(self, bitmap=dc.GetAsBitmap())
      self.image_ctrl.Bind(wx.EVT_LEFT_UP, self.onImage)
      self.sizer.Add(self.image_ctrl, 0, wx.CENTER)

      # Part for displaying the result
      self.button = wx.Button(self, label='Stylize', size=(256, 40))
      self.button.Bind(wx.EVT_BUTTON, self.onStylize)
      self.sizer.Add(self.button, 0, wx.CENTER)

      self.SetSizer(self.sizer)
      self.SetAutoLayout(1)
      self.sizer.Fit(self)

      self.makeMenuBar()
      self.Center()

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

    def onAbout(self, e):
      dlg = wx.MessageDialog(self, "App for doing style transfers", "About Style Transfer")
      dlg.ShowModal()
      dlg.Destroy()

    def onExit(self, e):
      self.Close()

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


if __name__ == '__main__':
  app = wx.App()
  frame = MainWindow(None, "Style Transfer")
  frame.Show()
  app.MainLoop()
