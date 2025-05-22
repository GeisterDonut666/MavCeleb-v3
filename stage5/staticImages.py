"""
M. Saad Saeed
18F-MS-CP-01
"""


#from google_images_download import google_images_download
from simple_image_download import simple_image_download as simp
import shutil
import glob
import os
import cv2


n_images = 500


startDown = simp.simple_image_download
try:
    shutil.rmtree('staticImages')
except:
    print('Directory already deleted')

# Read identities
names = []
with open('idList','r+') as idList:
    for n in idList:
        n = n.split('\t')[0]
        names.append(n)

# Download images
for n in names:
    print(n)
    startDown().download(keywords=str(n.split('\t')[0]), limit=n_images)

# Remove corrupted images
for fold in glob.glob(os.path.join('simple_images', '*')):
    for img in glob.glob(os.path.join(fold, '*')):
        im = cv2.imread(img)
        if im is None:
            os.remove(img)
            #print('corrupted image removed')

# Rename images to iterating numbers
for fold in glob.glob(os.path.join('simple_images', '*')):
    print(fold)
    print(len(glob.glob(os.path.join(fold, '*.jpg'))))
    for img, index in zip(glob.glob(os.path.join(fold, '*.jpg')), range(len(glob.glob(os.path.join(fold, '*.jpg'))))):
        im = cv2.imread(img)
        cv2.imwrite(os.path.join(fold, 'im_{:04d}.jpg'.format(index)), im)
        os.remove(img)
