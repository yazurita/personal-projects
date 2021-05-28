#from astropy.io import fits
#import degrade.psf as PSF


import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import time
from tqdm import tqdm
import argparse
from modules.Model import TransformerEncoder
import scipy.io as io

try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False

# https://github.com/jadore801120/attention-is-all-you-need-pytorch

def optimal_binning(inp, nbins):
    
      tmp = ECDF(inp)

      xnew = np.linspace(0.0, 1.0, nbins+1)[0:-1]
      out = np.interp(xnew, tmp.y, tmp.x)

      return out

class Dataset(torch.utils.data.Dataset):

    def __init__(self):
        super(Dataset, self).__init__()

        data = np.load('data_prep/data.npy', allow_pickle=True).item()
            
        self.stokes, self.mags, self.norm_m, self.norm_s, self.n_training, self.tau, self.wavelength = data.values()
        self.n_mags = self.mags.shape[0]
        self.n_stokes = self.stokes.shape[0]
        
                                
    def __getitem__(self, index):
        mags = [self.mags[i][index, :][:, None] for i in range(self.n_mags)]
        mags = np.concatenate(mags, axis=-1)

        stokes = [self.stokes[i][index, :][:, None] for i in range(self.n_stokes)]        
        stokes = np.concatenate(stokes, axis=-1)

        mask = self.wavelength <= 0
            
        return stokes.astype('float32'), self.wavelength.astype('float32'), mags.astype('float32'), mask, self.tau.astype('float32')
      
    def __len__(self):
        return self.n_training



class Training(object):
    def __init__(self, batch_size, gpu=0, validation_split=0.2):
        
        # Get device and batch size 
        self.gpu = gpu
        self.batch_size = batch_size
        self.device = torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() else 'cpu')                
        
        if (self.device != 'cpu'):
            if (NVIDIA_SMI):
                nvidia_smi.nvmlInit()
                self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
                print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))

        # Model hyperparameters that we can tune
        self.hyperparameters = {
            'n_input' : 4,                        # IQUV
            'embed_dim' : 64,                     # Embedding dimension for the Transformer Encoder (128 is probably too large)
            'num_heads' : 8,                      # Number of heads of the Encoder
            'num_layers' : 6,                     # Number of layers of the Encoder
            'ff_dim' : 64*4,                      # Dimension of the internal fully connected networks in the Encoder (should be divisible by the number of heads)
            'dropout' : 0.1,                      # Dropout in the Encoder
            'norm_in' : True,                     # Order of the Layer Norm in the Encoder (always True for good convergence using PreNorm)
            'latent_dim' : 64,                    # Latent dimension that enters the SIREN
            'num_siren_layers' : 3,               # Number of SIREN layers
            'weight_init_type' : 'xavier_normal', # Initialization of the Encoder
            'warmup' : 'no',                      # Use learning rate warmup during training (if PreNorm, not really necessary)
            'n_warmup_steps' : 40000              # Number of warmup steps if it applies
        }
        
        # Define encoder and decoder
        self.model = TransformerEncoder(self.hyperparameters).to(self.device)
        
        # Load the dataset
        self.dataset = Dataset()
        
        self.scale_model = torch.tensor(self.dataset.norm_m.astype('float32')).to(self.device)
        
        # Shuffle the dataset and split in training+validation
        idx = np.arange(self.dataset.n_training)
        np.random.shuffle(idx)

        self.train_index = idx[0:int((1-validation_split)*self.dataset.n_training)]
        self.validation_index = idx[int((1-validation_split)*self.dataset.n_training):]

        # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)

        self.train_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size = self.batch_size, shuffle = False)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.validation_sampler, batch_size = self.batch_size, shuffle = False)            
              
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def _get_lr_scale(self, epoch):
        """
        Function to be called by the scheduler to compute the learning rate if warmup is used or not
        """
        if (self.hyperparameters['warmup'] == 'yes'):                
            d_model = self.hyperparameters['embed_dim']
            n_warmup_steps = self.hyperparameters['n_warmup_steps']
            n_steps = epoch + 1
            lr = (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))
        else:
            lr = 1.0

        return lr

    def init_optimize(self, lr, smooth):
        """
        Initialization
        
        """

        # Initial learning rate
        self.lr = lr

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08)
            
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._get_lr_scale, last_epoch=-1)

        # Loss function
        self.loss_fn = nn.MSELoss()

        self.smooth = smooth

        current_time = time.strftime("%Y-%m-%d-%H:%M")
        self.out_name = f'weights/{current_time}.pth'

        print(f"Output : {self.out_name}")

    def optimize(self, epochs):
        """
        Optimization of the model
        """
        
        CLIP = 1

        self.n_epochs = epochs

        best_valid_loss = float('inf')
        tloss_ls = []
        vloss_ls = []

        for epoch in range(epochs):
            
            start_time = time.time()
            
            # Call one training epoch and then validation
            train_loss = self.train(CLIP, epoch=epoch)
            valid_loss = self.validate(epoch=epoch)
            tloss_ls.append(train_loss)
            vloss_ls.append(valid_loss)

            end_time = time.time()
                        
            # Save model if better validation loss
            if valid_loss < best_valid_loss:
                print("Saving model...")
                best_valid_loss = valid_loss
                
                checkpoint = {'epoch': epoch,
                    'lr': self.lr,
                    'state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': tloss_ls,
                    'val_loss': vloss_ls,                    
                    'hyperparameters': self.hyperparameters}
                
                torch.save(checkpoint, f'{self.out_name}')
                        
    
    def train(self, clip, epoch):
        """
        Train one epoch
        """
        
        self.model.train()
        
        t = tqdm(self.train_loader)
                
        loss_avg = 0.0
        accuracy = 0

        n_word_total, n_word_correct = 0, 0

        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
        
        for i, (stokes, wavelength, model, mask, tau) in enumerate(t):
            
            # Move data to computing device
            stokes, wavelength, model, mask, tau = stokes.to(self.device), wavelength.to(self.device), model.to(self.device), mask.to(self.device), tau.to(self.device)

            # Zero all gradients
            self.optimizer.zero_grad()
            
            # Call model
            output = self.model(wavelength, stokes, mask, tau)

            # Compute loss
            loss = self.loss_fn(output, model)

            # Compute standard deviation of model-target
            std = (torch.std(output - model, dim=(0, 1)) * self.scale_model).detach().cpu().numpy()
                            
            # Backpropagate
            loss.backward()
            
            # Clip gradients to avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            
            # Optimizer step
            self.optimizer.step()

            # Scheduler step
            self.scheduler.step()

            # Get current learning rate
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

            # Show status
            if (i == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                        
            if (NVIDIA_SMI):
                usage = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                t.set_postfix(loss=loss_avg, dT=f'{std[0]:5.1f}', dBz=f'{std[1]:5.1f}', dv=f'{std[-1]/1e5:5.1f}', epoch=f'{epoch:3d}/{self.n_epochs:3d}', lr=current_lr, gpu=usage.gpu, memused=f'{memory.used/1024**2:5.1f} MB')
            else:
                t.set_postfix(loss=loss_avg, epoch=f'{epoch}/{self.n_epochs}')
        
        return loss_avg

    #function to train model
    def validate(self, epoch):
        
        self.model.eval()
        with torch.no_grad():
        
          t = tqdm(self.validation_loader)
                  
          loss_avg = 0.0
          acc = 0
          n_word_total, n_word_correct = 0, 0
          
          for i, (stokes, wavelength, model, mask, tau) in enumerate(t):
                          
              stokes, wavelength, model, mask, tau = stokes.to(self.device), wavelength.to(self.device), model.to(self.device), mask.to(self.device), tau.to(self.device)
                          
              output = self.model(wavelength, stokes, mask, tau)

              loss = self.loss_fn(output, model)
              
              if (i == 0):
                  loss_avg = loss.item()
              else:
                  loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                          
              t.set_postfix(acc=f'{acc:5.1f}', loss=loss_avg, epoch=f'{epoch}/{self.n_epochs}')
          
          return loss_avg


if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='Learning rate')
    parser.add_argument('--gpu', '--gpu', default=0, type=int,
                    metavar='GPU', help='GPU')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float,
                    metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--epochs', '--epochs', default=200, type=int,
                    metavar='EPOCHS', help='Number of epochs')
    parser.add_argument('--batch', '--batch', default=128, type=int,
                    metavar='BATCH', help='Batch size')
    
    parsed = vars(parser.parse_args())

    deepnet = Training(batch_size=parsed['batch'], gpu=parsed['gpu'])

    deepnet.init_optimize(lr=parsed['lr'], smooth=parsed['smooth'])
    deepnet.optimize(parsed['epochs'])
