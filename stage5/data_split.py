import glob
import os
from random import random, seed
from shutil import copy


# NOTE: Careful with changing the filepaths, I think you will need to change multiple strings for it to work as intended


def split_data(data_dir: str, val_ratio: float=0.2) -> None:
    subdirs = ['train_test_split/train/', 'train_test_split/valid/']
    for subdir in subdirs:
        for labeldir in glob.glob(data_dir + '/*'):
            newdir = subdir + labeldir.split('/')[1]
            try:
                os.makedirs(newdir)
            except:
                print('Already Created')
    seed(1)
    for file in os.listdir(data_dir):
        for images in os.listdir(data_dir + '/' + file):
            src = data_dir + '/' + file + '/' + images
            #            print(src)
            dst_dir = 'train_valid_split/train/'
            ran = random()
            if ran < val_ratio:
                dst_dir = 'train_valid_split/valid/'
            if file.startswith(file):
                dst = dst_dir + file
                print(dst)
                copy(src, dst)


data_dir = 'staticFaceExtractedClustered'    #
if not os.path.exists(data_dir + 'train_valid_split', val_ratio=0.2):
    split_data(data_dir)
else:
    print('Already Split')
