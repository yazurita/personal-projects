# -*- coding: utf-8 -*-

import torch
import torchvision.models as models
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from IPython.display import HTML, clear_output
import PIL.Image

def resize(img, size, resample):
  img_01 =(img-np.min(img))/(np.max(img)-np.min(img))
  img_uint8 = np.uint8(img_01*255)
  img_pil = PIL.Image.fromarray(img_uint8)
  img_rzs = img_pil.resize(size, resample)
  img_rzs_np = np.float32(img_rzs)/255.0
  return img_rzs_np

def to_nchw(img):
  img = torch.as_tensor(img)
  img = img[...,None].repeat(1,1,3)
  img = img[None,...]
  return img.permute(0, 3, 1, 2)

def to_rgb(x):
  return x[...,:3,:,:]+0.5

class Style():
  def __init__(self, path):
    self.path = path
    self.img = fits.open(self.path)[0].data.astype(np.float32)

  def info(self):
    with fits.open(self.path) as hdul:
      hdul.info()
  
  def plot(self, color="viridis", colorbar = True):
    plt.imshow(self.img, cmap=color)
    if colorbar:
      plt.colorbar()
  
  def get_img(self):
    return self.img

class FeatureResponses():
  def __init__(self, model, style_layers):
    self.model = model
    self.style_layers = style_layers

  def norm_image(self, img):
    mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
    std = torch.tensor([0.229, 0.224, 0.225])[:,None,None]
    x = (img-mean) / std
    return x

  def correlations(self, img):
    x = self.norm_image(img)
    grams = []
    for i, layer in enumerate(self.model[:max(self.style_layers)+1]):
      x = layer(x)
      if i in self.style_layers:
        h, w = x.shape[-2:]
        y = x.clone()  # workaround for pytorch in-place modification bug
        gram = torch.einsum('bchw, bdhw -> bcd', y, y) / (h*w)
        grams.append(gram)
    return grams

  def style_loss(self, grams_x, grams_y):
    loss = 0.0
    for x, y in zip(grams_x, grams_y):
      loss += (x-y).square().mean()
    return loss

# model
class CA(torch.nn.Module):
  def __init__(self, chn, n_hidden, kernels):
    super().__init__()
    self.chn = chn
    self.w1 = torch.nn.Conv2d(chn*4, n_hidden, 1)
    self.w2 = torch.nn.Conv2d(n_hidden, chn, 1, bias=False)
    self.w2.weight.data.zero_()
    self.kernels = kernels

  # per channel convolution
  def perception(self, x):
    # batch, channel, height, width
    b, ch, h, w = x.shape
    y = x.reshape(b*ch, 1, h, w)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
    y = torch.nn.functional.conv2d(y, self.kernels[:,None])
    # returns perception vector (no channels anymore)
    return y.reshape(b, -1, h, w)

  def forward(self, x, update_rate=0.5):
    y = self.perception(x)
    y = self.w2(torch.relu(self.w1(y)))
    b, c, h, w = y.shape
    
    #stochastic update
    udpate_mask = (torch.rand(b, 1, h, w)+update_rate).floor()
    return x+y*udpate_mask

  def seed(self, n, sz):
    return torch.zeros(n, self.chn, sz, sz)

class Trainer():
  def __init__(self, opt, lr_sched, ca, n_pool, fr, style_img):
    self.opt = opt
    self.lr_sched = lr_sched
    self.ca = ca
    self.n_pool = n_pool
    self.fr = fr
    
    self.style_img_sz = style_img.shape[0]
    with torch.no_grad():
      self.target_grams = self.fr.correlations(to_nchw(style_img))
      self.pool = ca.seed(n_pool, self.style_img_sz)

  def epoch(self, curr_epoch):
    with torch.no_grad():
      batch_idx = np.random.choice(self.n_pool, 4, replace=False)
      x = self.pool[batch_idx]
      if curr_epoch % 2 == 0:    # every second batch contains the seed
        x[:1] = self.ca.seed(1, self.style_img_sz)
    n_steps = np.random.randint(32, 96)
    
    for k in range(n_steps):
      x = self.ca(x)
    imgs = to_rgb(x)
    grams_x = self.fr.correlations(imgs)
    loss = self.fr.style_loss(grams_x, self.target_grams)
    with torch.no_grad():
      loss.backward()

      for p in self.ca.parameters():
        p.grad /= (p.grad.norm()+1e-8)   # normalize gradients 
      self.opt.step()
      self.opt.zero_grad()
      self.lr_sched.step()
      self.pool[batch_idx] = x                      # update pool

    return x, loss, self.lr_sched.get_lr()[0]
  
  def train(self, n_epochs):
    loss_log = []
    for i in range(n_epochs):
      x, loss, lr = self.epoch(i)
      with torch.no_grad():
        loss_log.append(loss.item())
        if i%100==0:
          clear_output(True)
          plt.plot(range(len(loss_log)), loss_log, '.', alpha=0.1)
          plt.yscale('log')
          plt.ylim(np.min(loss_log), loss_log[0])
          plt.show()
          imgs = to_rgb(x).permute([0, 2, 3, 1]).cpu()
          plt.imshow(np.hstack(imgs))
          plt.show()
        if i%10 == 0:
          print('\rstep_n:', len(loss_log),
            ' loss:', loss.item(), 
            ' lr:', lr, end='')

#--- parameters
STYLE = Style(path='____.fits')
STYLE_IMG = STYLE.get_img()
STYLE.info(), STYLE.plot(color='gray')

MODEL = models.vgg16(pretrained=True).features
STYLE_LAYERS = [1, 6, 11, 18, 25]
FR = FeatureResponses(MODEL, STYLE_LAYERS)

# kernels: identity 3x3 matrix, sobel operator and laplacian matrix
ident = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
sobel_x = torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])/8.0
lap = torch.tensor([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])/16.0

CHN = 12
N_HIDDEN = 96
KERNELS = torch.stack([ident, sobel_x, sobel_x.T, lap])
CA = CA(CHN, N_HIDDEN, KERNELS)

OPTIMIZER = torch.optim.Adam(CA.parameters(), 1e-3)
LR_SCHED = torch.optim.lr_scheduler.MultiStepLR(OPTIMIZER, [2000], 0.3)

N_POOL = 1024
IMG_SIZE = 128
RESAMPLE = PIL.Image.LANCZOS
STYLE_IMG_RSZ = resize(STYLE_IMG, (IMG_SIZE,IMG_SIZE), RESAMPLE)
TRAINER = Trainer(OPTIMIZER, LR_SCHED, CA, N_POOL, FR, STYLE_IMG_RSZ)
#--- parameters

# train & save
TRAINER.train(4000)
torch.save(CA.state_dict(), '____')

#--- run this block to load saved weights
CHN = 12
N_HIDDEN = 96
ident = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
sobel_x = torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])/8.0
lap = torch.tensor([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])/16.0
KERNELS = torch.stack([ident, sobel_x, sobel_x.T, lap])
model = CA(CHN, N_HIDDEN, KERNELS)
model.load_state_dict(torch.load('____'))
#---

# create video
fig, ax = plt.subplots(figsize=(5.5,5.5))
plt.close()
imgs = []
with torch.no_grad():
  x = model.seed(n=1, sz=128)
  for f in range(750):
    x[:] = model(x)
    img = np.asarray((x[0,0,:,:]+0.5).cpu())
    img = resize(img, (1152, 1152), PIL.Image.LANCZOS)
    imgs.append([ax.imshow(img*255, cmap='gray',animated=True, vmax=255)])

anim = animation.ArtistAnimation(fig, imgs, interval=1000/30 , blit=False)
HTML(anim.to_html5_video())
