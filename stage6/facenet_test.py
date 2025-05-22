from pickle import load
import tensorflow as tf
import torchvision.io

import facenet
import numpy as np
import cv2
from math import ceil

from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, \
    training  # This line may show errors, but as long the imports run, they can be ignored
import glob
import os
from os import path, makedirs, remove
from PIL import Image
import torch;

print(torch.cuda.is_available())
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

save_path = 'facetracks_checked'
load_path = 'facetracks'

batch_size = 64
x = 224
y = 224

# Setup hardware
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
n_workers = 0 if os.name == 'nt' else 8
print('Running on device: {}'.format(device))


def get_total_videos(path: str):
    for ids in glob.glob(os.path.join(load_path, '*')):
        if not os.path.exists(save_path + '/' + ids.split('/')[1]):
            makedirs(save_path + '/' + ids.split('/')[1])

    vid_paths = glob.glob(os.path.join(path, '**', '*.avi'), recursive=True);
    vid_paths.sort()
    return vid_paths, len(vid_paths)


# Load finetuned model - insert whichever works best for you
model = torch.load(os.path.join('models', 'facenet_finetuned_adam_6.pt')).to(device)
model.to(device)
model.eval()

# Get a softmax function to turn prediction logits into probabilities
softmax = torch.nn.Softmax()

# Get number of facetracks
_, total_vids = get_total_videos(load_path)
print(f'Total number of videos: {total_vids}')

# Get class names dictionary
valid_dir = os.path.join('train_test_split_aug', 'valid')
valid_data = datasets.ImageFolder(valid_dir, transform=None)
class_names = new_dict = dict([(value, key) for key, value in valid_data.class_to_idx.items()])     # swap keys and values
print(class_names)

ids_paths = glob.glob(os.path.join(load_path, '*')); ids_paths.sort()
for label_true, ids in enumerate(ids_paths[5:]):
    for typ in glob.glob(os.path.join(ids, '*')):
        print(typ)
        for track_nr, links in enumerate(glob.glob(os.path.join(typ, '*', '*.avi'))[:5]):

            sp = links.split('/')
            #print(sp)

            # Create target folder
            if not os.path.exists(save_path + '/' + sp[1] + '/' + sp[2] + '/' + sp[3]):
                makedirs(save_path + '/' + sp[1] + '/' + sp[2] + '/' + sp[3])

            # Capture facetrack images
            cap = cv2.VideoCapture(links)
            tframes = int(cap.get(7))

            allframes = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = frame[32:x - 32, 32:y - 32]
                cv2.imwrite(
                    save_path + '/' + sp[1] + '/' + sp[2] + '/' + sp[3] + f'/track{track_nr}_' + '%04d.jpg' % (tframes),
                    frame)
                tframes = tframes - 1

            image_paths = glob.glob(os.path.join(save_path, sp[1], sp[2], sp[3], f'track{track_nr}_*.jpg'),
                                    recursive=True);
            image_paths.sort()

            nrof_images = len(image_paths)
            nrof_batches_per_epoch = int(ceil(1.0 * nrof_images / batch_size))
            #emb_array = np.zeros((nrof_images, embedding_size))

            #print(image_paths)
            #print(nrof_images, nrof_batches_per_epoch)
            #print(emb_array)

            predictions = []
            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = image_paths[start_index:end_index]

                images_batch = []
                for path in paths_batch:
                    images_batch.append(fixed_image_standardization(torchvision.io.read_image(path)))

                images_batch = torch.stack(images_batch, dim=0).to(torch.float).to(device)
                #images_batch.to(device)
                #images = facenet.load_data(paths_batch, False, False, 160)
                #feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                #emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                # Get predictions
                pred = model(images_batch)
                pred = softmax(pred)
                pred = pred.cpu().detach()
                predictions.append(pred)

            predictions = torch.cat(predictions, dim=0).numpy()
            best_class_ind = np.argmax(predictions, axis=1)

            #print(np.sum(predictions, axis=1))
            #print(best_class_ind)
            best_class_prob = predictions[np.arange(len(best_class_ind)), best_class_ind]
            acc = np.mean(np.equal(best_class_ind, np.ones_like(best_class_ind) * label_true))

            for im in glob.glob(os.path.join(save_path, '**', '*.jpg'), recursive=True):
                remove(im)



            with open(save_path + '/' + sp[1] + '/' + sp[2] + '/' + sp[3] + '/' + sp[4].split('.')[0] + '.txt',
                      'w+') as file:
                for i in range(len(best_class_ind)):
                    file.write('%4d  %s: %.3f\n' % (i, class_names[best_class_ind[i]], best_class_prob[i]))
            with open(save_path + '/' + sp[1] + '/' + sp[2] + '/' + sp[3] + '/' + sp[4].split('.')[0] + '.txt',
                      'a+') as file:
                file.write('Accuracy: %.3f' % acc)
            total_vids = total_vids - 1

        print('Remaining vids: ', total_vids)

