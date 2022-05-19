import sys
sys.path.insert(1, '.')
from model_utils import *
from KamitaniData.kamitani_data_handler import kamitani_data_handler
from torch.utils.data import DataLoader, Dataset
from imageio import imread
from PIL import Image
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
import pickle


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

sub = 3



with open(f"saved_regression_models/Subject{sub}_regressionanalysis_instancevector2048dim.p","rb") as f:
  datadict = pickle.load(f)
  W_instance = datadict['coef_']
  b_instance = datadict['intercept_']

with open(f"saved_regression_models/Subject{sub}_regressionanalysis_noiseZ_119dim.p","rb") as f:
  pickle.dump(datadict,f)
  W_noise = datadict['coef_']
  b_noise = datadict['intercept_']

with open(f"saved_regression_models/Subject{sub}_regressionanalysis_dense24576dim.p","rb") as f:
  pickle.dump(datadict,f)
  W_dense = datadict['coef_']
  b_dense = datadict['intercept_']
num_c, num_hw = 1536, 4

print('Regression Models are loaded')


kamitani_data_mat = './KamitaniData/fmri_data/Subject{}.mat'.format(sub)
test_img_csv = './KamitaniData/imageID_test.csv'
train_img_csv = './KamitaniData/imageID_training.csv'
handler = kamitani_data_handler(matlab_file = kamitani_data_mat, test_img_csv =test_img_csv, train_img_csv =train_img_csv)
Y,Y_test,Y_test_avg = handler.get_data(normalize = 1, roi = 'ROI_VC')
labels_train, labels = handler.get_labels()
num_voxels, num_train, num_test = Y.shape[1], 1200, 50

Y_sorted = np.zeros_like(Y)
for id in range(len(Y)):
  idx = (np.abs(id - labels_train)).argmin()
  Y_sorted[id] = Y[idx]

print('fMRI data is loaded')

ffa_voxs = handler.get_meta_field('ROI_FFA').astype(bool)
loc_voxs = handler.get_meta_field('ROI_LOC').astype(bool)
intersect_rois = ffa_voxs * loc_voxs
ppa_voxs = handler.get_meta_field('ROI_PPA').astype(bool)

rois = ['ROI_V1','ROI_V2','ROI_V3','ROI_V4']

roi_masks = []
all_out = []
for roi in rois:
    roi_masks.append(handler.get_meta_field(roi).astype(bool))
roi_masks.append(loc_voxs ^ intersect_rois)
roi_masks.append(ffa_voxs ^ intersect_rois)
roi_masks.append(ppa_voxs)

for roi_mask in roi_masks:
    roi_maximizer = np.zeros((Y_sorted.shape[1]))

    in_roi = roi_mask
    out_roi = np.invert(in_roi)
    roi_maximizer[in_roi] = 1
    roi_maximizer[out_roi] = 0




    roi_instance = roi_maximizer.reshape(1,-1) @ W_instance.T
    
    roi_instance /= np.linalg.norm(roi_instance)
    roi_instance = roi_instance + b_instance
    roi_instance /= np.linalg.norm(roi_instance)

    roi_noise = roi_maximizer.reshape(1,-1) @ W_noise.T + b_noise
    roi_dense = (roi_maximizer.reshape(1,-1) @ W_dense.T + b_dense).reshape(1, num_c, num_hw, num_hw)



    out = generate_rec(roi_instance.astype(np.float32),roi_noise.astype(np.float32),roi_dense.astype(np.float32))
    all_out.append(out)

all_out = np.concatenate(all_out)
plt.imsave('reconstructed_images/ROIMaxs_Sub{}.png'.format(sub),imgrid(image_to_uint8(all_out), cols=7,pad=5))
print('Finished')