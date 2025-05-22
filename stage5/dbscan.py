"""
M. Saad Saeed
18F-MS-CP-01
"""

import numpy as np
import facenet
from sklearn.cluster import DBSCAN
import tensorflow as tf
import os
import cv2
import align.detect_face
import glob


def create_network_face_detection(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def align_data(image_list, image_size, margin, pnet, rnet, onet):
    img_list = []
    for x in range(len(image_list)):
        #        aligned = cv2.resize(image_list[x], dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
        prewhitened = facenet.prewhiten(image_list[x])
        img_list.append(prewhitened)
        print(x)
    if len(img_list) > 0:
        images = np.stack(img_list)
        return images
    else:
        return None


os.environ["CUDA_VISIBLE_DEVICES"] = "5"
gpu_memory_fraction = 0
np.load.__defaults__ = (None, False, True, 'ASCII')
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
pnet, rnet, onet = create_network_face_detection(gpu_memory_fraction)
np.load = np_load_old
margin = 44
image_size = 160


def fdbscan(matrix, ids, image_list):
    db = DBSCAN(eps=0.78, min_samples=1, metric='precomputed')
    db.fit(matrix)
    labels = db.labels_
    no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    out_dir = 'out_dir/' + ids.split('/')[1]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    largest_cluster_only = 1
    print('No of clusters:', no_clusters)
    if no_clusters > 0:
        if largest_cluster_only >= 1:
            largest_cluster = 0
            for i in range(no_clusters):
                if len(np.nonzero(labels == i)[0]) > len(np.nonzero(labels == largest_cluster)[0]):
                    largest_cluster = i
            print('Saving largest cluster (Cluster: {})'.format(largest_cluster))
            cnt = 1
            for i in np.nonzero(labels == largest_cluster)[0]:
                cv2.imwrite(os.path.join(out_dir, str(cnt) + '.jpg'), image_list[i])
                cnt += 1
        else:
            print('Saving all clusters')
            for i in range(no_clusters):
                cnt = 1
                print('Cluster {}: {}'.format(i, np.nonzero(labels == i)[0]))
                path = os.path.join(out_dir, str(i))
                if not os.path.exists(path):
                    os.makedirs(path)
                    for j in np.nonzero(labels == i)[0]:
                        cv2.imwrite(os.path.join(path, str(cnt) + '.jpg'), image_list[j])
                        cnt += 1
                else:
                    for j in np.nonzero(labels == i)[0]:
                        cv2.imwrite(os.path.join(path, str(cnt) + '.jpg'), image_list[j])
                        cnt += 1


with tf.Graph().as_default():
    with tf.compat.v1.Session() as sess:
        facenet.load_model('model')
        for ids in glob.glob('staticFaceExtracted/*/')[::]:
            print(ids)
            image_list = load_images_from_folder(ids)
            images = align_data(image_list, image_size, margin, pnet, rnet, onet)
            images_placeholder = sess.graph.get_tensor_by_name("input:0")
            embeddings = sess.graph.get_tensor_by_name("embeddings:0")
            phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)
            nrof_images = len(images)
            matrix = np.zeros((nrof_images, nrof_images))
            for i in range(nrof_images):
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                    matrix[i][j] = dist
            fdbscan(matrix, ids, image_list)
