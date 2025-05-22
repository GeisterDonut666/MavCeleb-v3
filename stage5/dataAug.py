"""
M. Saad Saeed
18F-MS-CP-01
"""
import os.path
from glob import glob
from os import listdir, path, makedirs
from cv2 import imread, imwrite
from imgaug import augmenters as iaa
from random import shuffle
import numpy as np

# Enter folder names here. Don't forget to augment both train and valid images.
in_dir = 'train_test_split/valid'
out_dir = 'train_test_split_aug/valid'

augDataLim = 500    # train: 400, valid: 100
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
points_sampler = iaa.RegularGridPointsSampler(n_cols=20, n_rows=40)
seq = iaa.Sequential([
    sometimes(iaa.Crop(px=(0, 32)))  # crop images from each side by 0 to 16px (randomly chosen)
    , sometimes(iaa.GaussianBlur(sigma=(0, 3.0)))  # blur images with a sigma of 0 to 3.0
    , sometimes(iaa.Voronoi(points_sampler))
    , sometimes(iaa.Fliplr(0.1))
], random_order=True)

i = 0
print(glob(os.path.join(in_dir, '*')))
for ids in glob(os.path.join(in_dir, '*')):
    img = []
    lenCur = len(listdir(ids))
    print(ids)
    for im in glob(ids + '/*'):
        imTemp = imread(im)
        img.append(imTemp)
    arr = np.array(img)
    if not path.exists(out_dir + ids.split('/')[2]):
        makedirs(out_dir + ids.split('/')[2])
    for i in range(lenCur):
        imwrite(out_dir + ids.split('/')[2] + '/' + '%04d' % i + '.jpg', img[i])
    print('Data Copied')
    i += 1
    img_aug = seq(images=arr)
    flag = False  #

    lenAugDir = len(listdir(out_dir + ids.split('/')[2]))
    print(lenAugDir)
    while True:
        for im in img_aug:
            lenAugDir = len(listdir(out_dir + ids.split('/')[2]))
            #print(lenAugDir)
            if lenAugDir > augDataLim:
                flag = True
                break
            imwrite(out_dir + ids.split('/')[2] + '/' + '%04d' % i + '.jpg', im)
            i += 1
        shuffle(seq)
        img_aug = seq(images=arr)
        if flag:
            print('Augmented Successfully')
            break
    print(lenAugDir)
