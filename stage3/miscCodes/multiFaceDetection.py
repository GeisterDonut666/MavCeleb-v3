"""
M. Saad Saeed
18F-MS-CP-01
"""
import tensorflow as tf
import os
import align.detect_face
import numpy as np
import cv2

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
def createNetwork():
    np.load.__defaults__=(None, False, True, 'ASCII')
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    gpu_memory_fraction = 1
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options,
                log_device_placement=True))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(
                sess, None)
    np.load.__defaults__=(None, False, True, 'ASCII')
    return pnet, onet, rnet
def faceDetect(img,pnet,onet,rnet):
    minsize = 100 
    threshold = [ 0.6, 0.7, 0.7 ]
    factor = 0.709 # scale factor
    margin = 32
    image_size = 160
    bb = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(
            img, minsize, pnet,
            rnet, onet, threshold, factor)
    if bounding_boxes.size>0:
        for (x1, y1, x2, y2, acc) in bounding_boxes:
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(x1 - margin / 2, 0)
            bb[1] = np.maximum(y1 - margin / 2, 0)
            bb[2] = np.minimum(x2 + margin / 2, img_size[1])
            bb[3] = np.minimum(y2 + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = cv2.resize(cropped, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
            return True,bb,scaled
    else:
        bb = [0,0,0,0]
        scaled =  np.zeros([100,100,3],dtype=np.uint8)
        scaled.fill(0)
        return False,bb,scaled