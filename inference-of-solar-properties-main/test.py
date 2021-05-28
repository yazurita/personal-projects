import numpy as np
import torch
import matplotlib.pyplot as pl
import glob
import os
from random import randint
import time
from modules.Model import TransformerEncoder
from train import Dataset

class Testing(object):

  def __init__(self, model='transformer', checkpoint = 'weights/2021-05-25-10_25.pth'):
    self.device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
        
    if (checkpoint is None):
      files = glob.glob('weights/*.pth')
      checkpoint = max(files, key=os.path.getmtime)

    print(f"=> loading checkpoint {checkpoint}")
        
    chkp = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    self.hyperparameters = chkp['hyperparameters']
    self.tloss_ls = chkp['train_loss']
    self.vloss_ls = chkp['val_loss']
    self.epoch = chkp['epoch']
      
    self.model = TransformerEncoder(self.hyperparameters).to(self.device)
        
    self.model.load_state_dict(chkp['state_dict'])
    print("=> checkpoint loaded")

    self.model.eval()

  def predict(self, *item):
    return self.model(*item)
    
if (__name__ == '__main__'):
  test = Testing(model='transformer', checkpoint = 'weights/2021-05-25-10_25.pth')
  device = test.device
  dataset = Dataset()
  norm_m = dataset.norm_m
  n_training = dataset.n_training
  z = np.arange(dataset.tau.shape[0])

  current_time = time.strftime("%Y-%m-%d-%H_%M")
  os.mkdir(f'results/{current_time}')
  
  mags_names = ['T [K]', r'P [dyn cm$^{-1}$]', '|Bx| [G]', r'vlos [cm s$^{-1}$]' ,'By [G]', 'Bz [G]']
  #mags_names = ['T', '|Bx|','v_los' ,'By', 'Bz']
  pl.rcParams['lines.markersize'] = 1.2
  pl.rcParams['lines.linewidth'] = 0.8
  #pl.rcParams['figure.subplot.left'] = 0.07
  #pl.rcParams['figure.subplot.right'] = 0.99
  #pl.rcParams['figure.subplot.bottom'] = 0.05
  #pl.rcParams['figure.subplot.top'] = 0.95
  pl.rcParams['figure.subplot.wspace'] = 0.5
  pl.rcParams['figure.subplot.hspace'] = 0.5
  pl.rcParams['font.size'] = 8
  pl.rcParams['axes.labelweight'] = 'bold'
  
  
  with torch.no_grad():
    for j in range(5):
      x = randint(0, n_training - 1)
      stokes, wavelength, mags, mask, tau = dataset[x]
      stokes, wavelength, mask, tau = [torch.tensor(elem).to(device)[None, :] for elem in [stokes, wavelength, mask, tau]]
      output = test.predict(wavelength, stokes, mask, tau).cpu()

      fig, axs = pl.subplots(3, 2)

      for i in range(6):
          
        if i == 1:
          axs[i//2, 0 if i%2==0 else 1].plot(z, 10**( output[:,i] * norm_m[i] ), 'r-')
          axs[i//2, 0 if i%2==0 else 1].plot(z, 10**( mags[:,i] * norm_m[i] ), 'k-')
        
        else:
          axs[i//2, 0 if i%2==0 else 1].plot(z, output[:, i] * norm_m[i], 'r-')
          axs[i//2, 0 if i%2==0 else 1].plot(z, mags[:,i] * norm_m[i], 'k-')
        """
        axs[i//2, 0 if i%2==0 else 1].plot(z, output[:,i] * norm_m[i], 'r-')
        axs[i//2, 0 if i%2==0 else 1].plot(z, mags[:,i] * norm_m[i], 'k-')
        """
        axs[i//2, 0 if i%2==0 else 1].set_ylabel(f'{mags_names[i]}')
      
      pl.tight_layout()
      fig.legend(labels=['model', 'target'], loc="upper right")
      pl.subplots_adjust(right=0.88)
      fig.savefig(f'results/{current_time}/{x}.png')
    

    pl.close('all')
    pl.plot(np.arange(test.epoch + 1), test.tloss_ls, 'ko')
    pl.plot(np.arange(test.epoch + 1), test.vloss_ls, 'ro')
    #pl.xlim(0,800)
    pl.legend(['train loss', 'valid loss'], fontsize='x-large')
    pl.xlabel('epoch')
    pl.savefig(f'results/{current_time}/loss.png')







