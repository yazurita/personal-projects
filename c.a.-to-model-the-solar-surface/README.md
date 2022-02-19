# C.A. to model the solar surface

## The project in a nutshell

We created a simulation of the movements of the sun’s surface using just this one image as input:
<p align="center">
  <img src="https://user-images.githubusercontent.com/72736453/119829039-b8fec700-bef2-11eb-954e-b0e85c13a3bd.jpg" width=30% height=30% />
</p>

To create a mutable, self-organizing texture that resembles the one given above, we rely on Neural Cellular Automata. The idea is that every pixel on an initially white canvas will obey the same rule, which will update their values in each iteration depending on their neighbours’. Such update rule is learned using Artificial Neural Networks.

Voilà! An artificial sun.
<p align="center">
  <img src="https://user-images.githubusercontent.com/72736453/119829065-c0be6b80-bef2-11eb-9749-208d4f222bb0.gif" width=30% height=30% />
</p>

For the sake of comparison, here we have our simulation (left) and a timelapse created using a conventional model (right):
<p align="center">
  <img src="https://user-images.githubusercontent.com/72736453/119829429-227ed580-bef3-11eb-83c2-0b03238e6dd1.gif" width=25% height=25% />
  <img src="https://user-images.githubusercontent.com/72736453/119829435-24489900-bef3-11eb-80d6-d89356052247.gif" width=25% height=25% />
</p>

## How the update rule is learned

Once the so-called “perception vector”, which gathers the information about the pixel and its surroundings, is calculated, it goes through a two-dimensional convolution layer, a ReLU activation and a second two-dimensional convolution layer. The architecture’s weights will be updated according to the difference between the texture extracted from the output and the one extracted from the target image.
(more on style capture [here](https://arxiv.org/abs/1508.06576); more about the model [here](https://distill.pub/selforg/2021/textures/))


## Future improvement

The model works with 128 x 128 px images. An increase in size would make even the 16 Gb GPU on Google Collaboratory run out of memory. The challenge then is to create a scalable model to preserve as much resolution as possible.

(Note that the target image had to be downscaled from 1152 x 1152 px to 128 x 128 px, and then, upscaled to the original size again. Thus, the quality loss is significant)
<p align="center">
  <img src="https://user-images.githubusercontent.com/72736453/119829604-50641a00-bef3-11eb-840a-1338782f9f13.png" width=70% height=70% />
</p>

## Notes

Project in collaboration with Andrés Asensio.
Code inspired by [Niklasson et al.](https://distill.pub/selforg/2021/textures/).
