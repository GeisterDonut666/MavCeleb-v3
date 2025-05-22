"""
M. Saad Saeed
18F-MS-CP-01
"""


import hashlib
import cv2
import os
import glob

folders = glob.glob('simple_images/*')    # Enter name of your image folder here
folders.sort()
print(folders)
  
for fold in folders[:]:
    print(fold)
    duplicates = []
    hash_keys = {}
    for img,index in zip(glob.glob(fold+'/*.jpg'),range(len(glob.glob(fold+'/*.jpg')))):
        if os.path.isfile(img):
            tempIm = cv2.imread(img)
            filehash = hashlib.md5(tempIm).hexdigest()
            if filehash not in hash_keys: 
                hash_keys[filehash] = index
            else:
                duplicates.append((index,hash_keys[filehash]))
    if len(duplicates) > 0:
        print('Duplicates found. Removing')
        images = glob.glob(fold+'/*')
        for i in range(len(duplicates)):
            os.remove(images[duplicates[i][0]])
            print('Removed: ',images[duplicates[i][0]])
        
        count=0
        temp = []
        print('Rewriting')
        images = glob.glob(fold+'/*')
        for im,index in zip(images,range(len(images))):
            temp.append(cv2.imread(im))
            os.remove(im)
    
        for i in range(index):
            cv2.imwrite(fold+'/{:03d}.jpg'.format(i),temp[i])
    else:
        print('No duplicate')