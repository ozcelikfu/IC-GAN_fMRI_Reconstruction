import numpy as np
import pandas as pd
import sklearn.linear_model as skl
from KamitaniData.kamitani_data_handler import kamitani_data_handler
import pickle


sub=3
print('Loading fMRI Data for Subject',sub)
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

## Instance Features Regression
print('Training Instance Feature Regression')

train_instance = np.load('./extracted_features/instance_features.npz')['train_instance']

reg_instance = skl.Ridge(alpha=500, max_iter=1000, fit_intercept=True)
reg_instance.fit(Y_sorted, train_instance)
pred_test_latent = reg_instance.predict(Y_test_avg)
std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
pred_instance = std_norm_test_latent * np.std(train_instance,axis=0) + np.mean(train_instance,axis=0)

datadict = {
    'coef_' : reg_instance.coef_,
    'intercept_' : reg_instance.intercept_,
    'n_iter_': reg_instance.n_iter_,
    'n_features_in_':reg_instance.n_features_in_,
    # 'feature_names_in_':reg_instance.features_names_in_

}

with open(f"saved_regression_models/Subject{sub}_regressionanalysis_instancevector2048dim.p","wb") as f:
  pickle.dump(datadict,f)

## Noise Vectors Regression
print('Training Noise Vector Regression')

train_noise = np.load('./extracted_features/noise_vectors.npy')

reg_noise = skl.Ridge(alpha=1, max_iter=1000, fit_intercept=True)
#reg = skl.Lasso(alpha=0.01, max_iter=1000, fit_intercept=True)
#reg = skl.RidgeCV(alphas=(0.1,1,10,100,1000), cv=10, fit_intercept=True)
reg_noise.fit(Y_sorted, train_noise)
pred_noise = reg_noise.predict(Y_test_avg)

datadict = {
    'coef_' : reg_noise.coef_,
    'intercept_' : reg_noise.intercept_,
    'n_iter_': reg_noise.n_iter_,
    'n_features_in_':reg_noise.n_features_in_,
    # 'feature_names_in_':reg_instance.features_names_in_

}

with open(f"saved_regression_models/Subject{sub}_regressionanalysis_noiseZ_119dim.p","wb") as f:
  pickle.dump(datadict,f)

## Dense Vectors Regression
print('Training Dense Vector Regression')

train_dense = np.load('./extracted_features/dense_vectors.npy')

num_train, num_c, num_hw = 1200, 1536, 4

reg_dense = skl.Ridge(alpha=100, max_iter=1000, fit_intercept=True)
#reg = skl.Lasso(alpha=0.01, max_iter=1000, fit_intercept=True)
#reg = skl.RidgeCV(alphas=(0.1,1,10,100,1000), cv=10, fit_intercept=True)
reg_dense.fit(Y_sorted, train_dense)
pred_dense = reg_dense.predict(Y_test_avg)
pred_dense = pred_dense.reshape(num_test, num_c, num_hw, num_hw)

datadict = {
    'coef_' : reg_dense.coef_,
    'intercept_' : reg_dense.intercept_,
    'n_iter_': reg_dense.n_iter_,
    'n_features_in_':reg_dense.n_features_in_,
    # 'feature_names_in_':reg_instance.features_names_in_

}

with open(f"saved_regression_models/Subject{sub}_regressionanalysis_dense24576dim.p","wb") as f:
  pickle.dump(datadict,f)

print('Saving predicted test features for Reconstruction')
np.savez('extracted_features/predicted_test_features_Sub{}.npz'.format(sub), pred_instance=pred_instance, pred_noise=pred_noise, pred_dense=pred_dense)

