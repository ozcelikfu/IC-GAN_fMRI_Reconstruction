# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:40:55 2021

@author: ozcelik
"""

import io
import IPython.display
import os
from pprint import pformat
import numpy as np
from torch.utils.data import DataLoader, Dataset
from imageio import imread
from PIL import Image
import h5py

def imgrid(imarray, cols=4, pad=1, padval=255, row_major=True):
  """Lays out a [N, H, W, C] image array as a single image grid."""
  pad = int(pad)
  if pad < 0:
    raise ValueError('pad must be non-negative')
  cols = int(cols)
  assert cols >= 1
  N, H, W, C = imarray.shape
  rows = N // cols + int(N % cols != 0)
  batch_pad = rows * cols - N
  assert batch_pad >= 0
  post_pad = [batch_pad, pad, pad, 0]
  pad_arg = [[0, p] for p in post_pad]
  imarray = np.pad(imarray, pad_arg, 'constant', constant_values=padval)
  H += pad
  W += pad
  grid = (imarray
          .reshape(rows, cols, H, W, C)
          .transpose(0, 2, 1, 3, 4)
          .reshape(rows*H, cols*W, C))
  if pad:
    grid = grid[:-pad, :-pad]
  return grid

def interleave(*args):
  """Interleaves input arrays of the same shape along the batch axis."""
  if not args:
    raise ValueError('At least one argument is required.')
  a0 = args[0]
  if any(a.shape != a0.shape for a in args):
    raise ValueError('All inputs must have the same shape.')
  if not a0.shape:
    raise ValueError('Inputs must have at least one axis.')
  out = np.transpose(args, [1, 0] + list(range(2, len(a0.shape) + 1)))
  out = out.reshape(-1, *a0.shape[1:])
  return out

def imshow(a, format='png', jpeg_fallback=True):
  """Displays an image in the given format."""
  a = a.astype(np.uint8)
  data = io.BytesIO()
  Image.fromarray(a).save(data, format)
  im_data = data.getvalue()
  try:
    disp = IPython.display.display(IPython.display.Image(im_data))
  except IOError:
    if jpeg_fallback and format != 'jpeg':
      print ('Warning: image was too large to display in format "{}"; '
             'trying jpeg instead.').format(format)
      return imshow(a, format='jpeg')
    else:
      raise
  return disp

def image_to_uint8(x):
  """Converts [-1, 1] float array to [0, 255] uint8."""
  x = np.asarray(x)
  x = (256. / 2.) * (x + 1.)
  x = np.clip(x, 0, 255)
  x = x.astype(np.uint8)
  return x

def pair_images(array):
  S,H,W,C = array.shape
  paired_images = np.zeros((S//2, H, 2*W, C))
  for idx in range(S//2):
    paired_images[idx, :, :W, :] = array[2*id



class batch_generator_external_images(Dataset):
    """
    Generates batches of images from a directory
    :param img_size: image should be resized to size
    :param batch_size: batch size
    :param ext_dir: directory containing images
    """
    def __init__(self, data_path = '/content/drive/My Drive/ResearchFiles/NeuralDecoding/images_256.npz', mode='test_images'):
        self.data_path = data_path
        self.data = np.load(self.data_path)
        self.mode = mode
        self.data = self.data[self.mode].astype(np.float32)


    def __getitem__(self,idx):
        img = self.data[idx]
        img = img*2 - 1
        return img

    def __len__(self):
        return  len(self.data)
