"""
M. Saad Saeed
18F-MS-CP-01
"""

import tensorflow as tf
import os
import align.detect_face
import cv2
import numpy as np
import glob
import shutil


in_dir = 'simple_images'
out_dir = 'extracted_images'

#os.environ["CUDA_VISIBLE_DEVICES"] = ""
gpu_memory_fraction = 1
print('Creating networks and loading parameters')
with tf.Graph().as_default():
    np.load.__defaults__=(None, False, True, 'ASCII')
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options,
            log_device_placement=True))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(
            sess, None)
        np.load = np_load_old
        parent = f'{in_dir}/'
        child = f'{out_dir}/'
        try:
            os.mkdir(child)
        except:
            shutil.rmtree(child)
        if not os.path.exists(child):
            os.mkdir(child)
        minsize = 50 
        threshold = [ 0.6, 0.7, 0.7 ]
        factor = 0.709
        margin = 44
        image_size = 160
        identityList = glob.glob(parent+'*/')
        identityList.sort()
        for identity in identityList[::]:
            fold1 = identity.split('/')[1]
            os.mkdir(child+fold1)
            idx = identity+'/*.jpg'
            images = glob.glob(idx)
            for img,count in zip(images,range(len(images))):
                tempImg = cv2.imread(img)
                img_size = np.asarray(tempImg.shape)[0:2]
                # Getting bounding boxes for faces
                # NOTE: For some images there's a weird behaviour, where a dimension is reduced to 0 and cannot be broadcasted for an operation. I don't know why, and don't want to know badly enoough. I value my sanity more than those few images, so I simply ignore them.
                try:
                    bounding_boxes, _ = align.detect_face.detect_face(
                            tempImg, minsize, pnet,
                            rnet, onet, threshold, factor)
                except:
                    continue
                # Cropping out faces
                for (x1, y1, x2, y2, acc) in bounding_boxes:
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(x1 - margin / 2, 0)
                    bb[1] = np.maximum(y1 - margin / 2, 0)
                    bb[2] = np.minimum(x2 + margin / 2, img_size[1])
                    bb[3] = np.minimum(y2 + margin / 2, img_size[0])
                    cropped = tempImg[bb[1]:bb[3], bb[0]:bb[2], :]
                    scaled = cv2.resize(cropped, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
                    savename = child+fold1+'/face_{:05d}'.format(count)+'.jpg'
                    cv2.imwrite(savename,scaled)
                    count += 1
                print(count)