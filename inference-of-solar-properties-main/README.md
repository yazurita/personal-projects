# Inference of solar properties

## The aim

We have trained a model to learn the underlying relationship between the polarized, multispectral light coming from the Sun described by a Stokes profile:

<p align="center">
  <img src="https://user-images.githubusercontent.com/72736453/119956965-8cea5100-bf99-11eb-9faa-462bffbe3df3.png" width=50% height=50% />
</p>

and the magnetic and thermodynamic state of its atmosphere:

<p align="center">
  <img src="https://user-images.githubusercontent.com/72736453/119957076-aa1f1f80-bf99-11eb-90b5-d3504b0f0f74.png" width=50% height=50% />
</p>


## Current results
<p align="center">
  <img src="https://user-images.githubusercontent.com/72736453/119957399-f36f6f00-bf99-11eb-9a8a-c988bb969cbb.png" width=45% height=45% />
  <img src="https://user-images.githubusercontent.com/72736453/119957400-f4080580-bf99-11eb-9d20-600ed268226b.png" width=45% height=45% />
</p>


## The arquitecture

A encoder-decoder network is used. The input passes through a [Transformer](https://arxiv.org/abs/1706.03762v5), which calculates a latent vector based on the most important properties of the data. This vector is then used to condition a implicit representation network (a [SIREN](https://arxiv.org/abs/2006.09661) in this case).

Besides being much faster than conventional inversion techniques, this model is conceived to achieve:

1. limitless height resolution at the output (conventional methods would retrieve values at certain heights, and then interpolate between the nodes to obtain the whole stratification).
2. independency from the input spectral range (although always within the one where the training set is defined).


## Data

Four cubes corresponding to the parameters I, Q, U V, each one containing 288 x 288 pixel images at 601 wavelenghts initially, and 112 after interpolating to recover values at the wavelengths at which the satellite Hinode measures. The spectral range is 630 - 633 nm, where the Fe I duplet is located.

The atmosphere state is characterized by the quantities temperature, preassure, line-of-sight velocity, and the three cartesian components of the magnetic field. Each one is represented by 288 x 288 pixel images at 61 heights.


## Notes

The idea for this project belongs to Andrés Asensio, who supervised my undergraduate dissertation and I am lucky to keep on collaborating with.

Implementations used:

- [Transformer](https://github.com/tnq177/witwicky)
- [SIREN](https://github.com/lucidrains/pi-GAN-pytorch/blob/main/pi_gan_pytorch/pi_gan_pytorch.py)
