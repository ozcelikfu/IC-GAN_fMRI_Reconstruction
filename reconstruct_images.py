from model_utils import *
from torch.utils.data import DataLoader, Dataset
from imageio import imread
from PIL import Image
import numpy as np
import h5py
import torch
import sys
import os
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=3)
args = parser.parse_args()
sub=args.sub

def dense_forward(z, feats, dense_vec):
  y = model.get_condition_embeddings(None, feats)
  # If hierarchical, concatenate zs and ys
  if model.hier:
      zs = torch.split(z, model.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
  else:
      ys = [y] * len(model.blocks)

  # First linear layer
  #h = model.linear(z)
  h = dense_vec
  # Reshape
  h = h.view(h.size(0), -1, model.bottom_width, model.bottom_width)

  # Loop over blocks
  for index, blocklist in enumerate(model.blocks):
      # Second inner loop in case block has multiple layers
      for block in blocklist:
          h = block(h, ys[index])

  # Apply batchnorm-relu-conv-tanh at output
  return torch.tanh(model.output_layer(h))

def generate_rec(input_features, noise, dense=None):
  with torch.no_grad():
    noise_vector = torch.tensor(noise.astype(np.float32), requires_grad=False, device='cuda')
    instance_vector = torch.tensor(input_features.astype(np.float32), requires_grad=False, device='cuda')
    if type(dense)==type(noise):
      dense_vec = torch.tensor(dense.astype(np.float32), requires_grad=False, device='cuda')
      out = dense_forward(noise_vector, instance_vector, dense_vec)
    else:
      out = get_output(noise_vector, None, instance_vector)
    out = out.detach().cpu().numpy().transpose((0,2,3,1))
    return out

experiment_name = 'icgan_biggan_imagenet_res256'
gen_model = 'icgan'
last_gen_model = None
model = None
model, last_gen_model = load_generative_model(gen_model, last_gen_model, experiment_name, model)
replace_to_inplace_relu(model)
eps = 1e-8

print('Model is loaded')

pred_features = np.load('extracted_features/predicted_test_features_Sub{}.npz'.format(sub))
pred_instance, pred_noise, pred_dense = pred_features['pred_instance'], pred_features['pred_noise'], pred_features['pred_dense']
print('Features are loaded')

print('Generating Images')
sub_out = []
for idx in range(50):
  out = generate_rec(pred_instance[idx:idx+1],pred_noise[idx:idx+1],pred_dense[idx:idx+1])
  sub_out.append(out)
sub_out = np.concatenate(sub_out) 
print(sub_out.shape)
plt.imsave('reconstructed_images/RecImages_Sub{}.png'.format(sub),imgrid(image_to_uint8(sub_out), cols=5,pad=5))
print('Finished')