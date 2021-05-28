import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from astropy.io import fits
import psf as PSF

class DataPreparation():
  def __init__(self):
    # Params : T, P, |B|, v_los, B_phi, B_theta
    #self.stokes_size = 4
    #self.model_size = 6

    # Read data cubes
    self.mags = fits.open('data_prep/ivan_models.fits')[0].data
    self.stokes = fits.open('data_prep/ivan_models_synth.fits')[0].data
  
    # Delete mags columns 0,1,4,5,6
    self.mags = np.delete(self.mags, [0,1,4,5,6,8], axis=0)

    # |B|, B_phi, B_theta -> |Bx|, By, Bz
    Bx = np.abs( self.mags[2] * np.sin(self.mags[4]) * np.cos(self.mags[5]) )
    By = self.mags[2] * np.sin(self.mags[4]) * np.sin(self.mags[5])
    Bz = self.mags[2] * np.cos(self.mags[4])
    self.mags[2], self.mags[4], self.mags[5] = Bx, By, Bz

    # P -> log10(P)
    self.mags[1] = np.log10(self.mags[1])

    # Reshape cubes and get their dimensions
    self.stokes = np.transpose(self.stokes, axes=(2,0,1,3))

    self.n_mags, self.nx, self.ny, self.nz = self.mags.shape
    self.n_stokes, self.nx, self.ny, self.n_lambda = self.stokes.shape


  def fill_nans_with_mean(self, stokes, n_stokes, n_lambda, nx):

    nan_idxs = list(zip(*np.where(np.isnan(stokes) == True)))
    nan_idxs = nan_idxs[::n_stokes * n_lambda]

    for s, x, y, l in nan_idxs:
      adj_rows = np.empty((2, n_stokes, n_lambda))
      adj_rows[0] = stokes[:, x-1, y,:]
      adj_rows[1] = stokes[:, x+1, y,:]
      stokes[:,x,y,:] = np.mean(adj_rows, axis=0)
    
    return stokes, nan_idxs


  def psf(self, stokes, n_stokes, n_lambda):
    stokes_psf = np.empty(stokes.shape)

    for i in np.arange(n_stokes):
      for j in np.arange(n_lambda):
        stokes_psf[i,:,:,j] = PSF.osys.convolve_with_psf(stokes[i,:,:,j])

    return stokes_psf


  def remove_nans(self, mags, stokes, nan_idxs):
    
    mags = mags.reshape((self.n_mags, self.nx*self.ny, self.nz))
    stokes = stokes.reshape((self.n_stokes, self.nx*self.ny, self.n_lambda))

    nan_idxs = [x * self.nx + y for s, x, y, l in nan_idxs]

    stokes = np.delete(stokes, nan_idxs, axis = 1)
    mags = np.delete(mags, nan_idxs, axis = 1)
    n_samples = mags.shape[1]

    return mags, stokes, n_samples
    

  def interpolate(self, stokes, n_stokes, n_lambda, n_samples, start = 6300.485, end = 6303.49):
    # Reduce n_lambda through interpolation
    lambdas = np.linspace(start, end, n_lambda)
    new_lambdas = np.loadtxt('data_prep/wavelengthHinode.txt')
    new_n_lambda = new_lambdas.shape[0]
    new_stokes = np.empty((n_stokes, n_samples, new_n_lambda))

    for i in range(n_stokes):
      for j in range(n_samples):
        tck = interpolate.splrep(lambdas, stokes[i,j,:])
        new_stokes[i, j, :] = interpolate.splev(new_lambdas, tck)
    
    return new_stokes, new_n_lambda
    
  def scale(self, mags, stokes, norm_m, norm_s):

    stokes /= norm_s[:, None, None]
    mags /= norm_m[:, None, None]
    
    return mags, stokes

  def forward(self):

    stokes, nan_idxs = self.fill_nans_with_mean(self.stokes, self.n_stokes, self.n_lambda, self.nx)

    stokes_psf = self.psf(stokes, self.n_stokes, self.n_lambda)
    
    mags, stokes, n_samples = self.remove_nans(self.mags, stokes_psf, nan_idxs)
    
    stokes, self.n_lambda = self.interpolate(stokes, self.n_stokes, self.n_lambda, n_samples)

    mean_I = np.mean(stokes[0,:,0])
    self.norm_s = np.array([mean_I, mean_I * 0.001, mean_I * 0.001, mean_I * 0.001])
    self.norm_m = np.array([1e4, 5.5, 5e3, 16e5, 5e3, 5e3])
    self.mags, self.stokes = self.scale(mags, stokes, self.norm_m, self.norm_s)

    self.n_training = n_samples
    self.tau = np.linspace(0, 1, self.nz)
    self.wavelength = np.linspace(0, 1, self.n_lambda)

    data = {'stokes': self.stokes,
            'mags': self.mags,
            'norm m': self.norm_m,
            'norm s': self.norm_s,
            'n training': self.n_training,
            'tau': self.tau,
            'wavelength': self.wavelength
            }
    np.save(f'data_prep/data.npy', data)



DataPreparation().forward()
