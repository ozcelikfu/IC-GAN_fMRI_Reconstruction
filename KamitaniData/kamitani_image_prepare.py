import numpy as np

import pandas as pd
from imageio import imread
import matplotlib.pyplot as plt
from skimage.transform import resize
import os


# Create Image dataset from Imagenet folders
def image_generate(imgnet_dir = './images/',test_csv='./imageID_test.csv',train_csv='./imageID_training.csv',size = 256,out_file= './images_256.npz',interpolation = 3):
    test_im = pd.read_csv(test_csv,header=None)
    train_im = pd.read_csv(train_csv,header=None)

    test_images = np.zeros([50, size, size, 3])
    train_images = np.zeros([1200, size, size, 3])

    count = 0

    for file in list(test_im[1]):
        img = imread(imgnet_dir + 'test' + '/' + file)
        test_images[count] = image_prepare(img, size,interpolation)
        count += 1

    count = 0

    for file in list(train_im[1]):
        img = imread(imgnet_dir + 'training' + '/' + file)
        train_images[count] = image_prepare(img, size,interpolation)
        count += 1
    np.savez(out_file, train_images=train_images, test_images=test_images)


#ceneter crop and resize
def image_prepare(img,size,interpolation):

    out_img = np.zeros([size,size,3])
    s = img.shape
    r = s[0]
    c = s[1]

    trimSize = np.min([r, c])
    lr = int((c - trimSize) / 2)
    ud = int((r - trimSize) / 2)
    img = img[ud:min([(trimSize + 1), r - ud]) + ud, lr:min([(trimSize + 1), c - lr]) + lr]

    img = resize(img, (size, size), order=interpolation)
    if (np.ndim(img) == 3):
        out_img = img
    else:
        out_img[ :, :, 0] = img
        out_img[ :, :, 1] = img
        out_img[ :, :, 2] = img

    return out_img/255.0

if __name__ == "__main__":
    print('creating npz file')
    image_generate()
