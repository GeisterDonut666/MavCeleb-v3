"""
M. Saad Saeed
18F-MS-CP-01
"""


from glob import glob
from pickle import load
import tensorflow as tf
import facenet
import numpy as np
import cv2
from os import path, makedirs, remove
from math import ceil


def totalVideos():
    tot=1
    for ids in glob('stage5/facetracks/*'):
        print(ids.split('/')[2]+'/'+ids.split('/')[3])
        if not path.exists(ids.split('/')[2]+'/'+ids.split('/')[3]):    # Why is this part even needed?
            makedirs(ids.split('/')[2]+'/'+ids.split('/')[3])
        for typ in glob(ids+'/*'):
            for links in glob(typ+'/*/*.avi'):
                tot+=1
    return tot


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
        batch_size = 256
        x = 224
        y = 224
        
        total_vids = totalVideos()
        print('Total Videos: ',total_vids)
        facenet.load_model('model')
        with open('classifier.pkl', 'rb') as infile:
            (model, class_names) = load(infile)
        
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        
        for ids in glob('../stage4/facetracks/*'):
            for typ in glob(ids+'/*'):
                for links in glob(typ+'/*/*.avi'):
                    
                    sp = links.split('/')
                    if not path.exists('finOut'+'/'+sp[3]+'/'+sp[4]+'/'+sp[5]):
                        makedirs('finOut'+'/'+sp[3]+'/'+sp[4]+'/'+sp[5])
                        cap = cv2.VideoCapture(links)
                        tframes = int(cap.get(7))
    
                        allframes = []
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame = frame[32:x-32,32:y-32]
                            cv2.imwrite(sp[2]+'/'+sp[3]+'/%04d.jpg'%(tframes), frame)
                            tframes = tframes-1
                            
                        dataset = facenet.get_dataset(sp[2])
                        paths, labels = facenet.get_image_paths_and_labels(dataset)
                        
                        nrof_images = len(paths)
                        nrof_batches_per_epoch = int(ceil(1.0*nrof_images / batch_size))
                        emb_array = np.zeros((nrof_images, embedding_size))
                        
                        for i in range(nrof_batches_per_epoch):
                            start_index = i*batch_size
                            end_index = min((i+1)*batch_size, nrof_images)
                            paths_batch = paths[start_index:end_index]
                            images = facenet.load_data(paths_batch, False, False, 160)
                            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                            emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                        
                        predictions = model.predict_proba(emb_array)
                        best_class_ind = np.argmax(predictions, axis=1)
                        best_class_prob = predictions[np.arange(len(best_class_ind)), best_class_ind]
                        acc = np.mean(np.equal(best_class_ind,labels))
                        for im in glob(sp[2]+'/'+sp[3]+'/*'):
                            remove(im)
                    
                    with open('finOut'+'/'+sp[3]+'/'+sp[4]+'/'+sp[5]+'/'+sp[6].split('.')[0]+'.txt','a+') as file:
                        for i in range(len(best_class_ind)):
                            file.write('%4d  %s: %.3f\n' % (i, class_names[best_class_ind[i]], best_class_prob[i]))
                    with open('finOut'+'/'+sp[3]+'/'+sp[4]+'/'+sp[5]+'/'+sp[6].split('.')[0]+'.txt','a+') as file:
                        file.write('Accuracy: %.3f' % acc)
                    total_vids = total_vids-1
                    print('Remaining vids: ', total_vids)
                    

confs = []
for test in glob('finOut/id0001/English/cFU_ez_naak/*.txt'):
    dat = []
    with open(test,'r+') as file:
        for d in file:
            d = d.split(':')[-1]
            dat.append(d)
        dat = np.asarray(dat,dtype='float64')
        av = dat[-1]
        if av<0.3:
            confs.append('0')
        else:
            dat = dat[0:-1]
            conf = ((np.median(dat)-min(dat))/len(dat))*100
            confs.append([test.split('/')[-1],conf])
#        else:
#            confs.append('0')
#    for txt in            

#dat = []
#with open('finOut/id0001/English/A8Ch7gJalls/00000.txt','r+') as file:
#    for d in file:
#        d = d.split(':')[-1]
#        dat.append(d)
#
#dat = np.asarray(dat,dtype='float64')
#dat = dat[0:-1]
#conf = np.zeros(len(dat))
#for i in range(len(dat)):
#    conf[i] = dat[i]/sum(dat)
#conf = (1-(dat/2))
#conf = np.median(dat)-min(dat)
        