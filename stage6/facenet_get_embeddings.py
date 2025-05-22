import glob
import os

from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
from PIL import Image
import torch; print(torch.cuda.is_available())


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


# Load MTCNN image preprocessing pipeline
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def read_into_array(paths: str, embedding) -> (np.array, np.array):
    """
    :param paths: Filepath of the images loaded. Expects subfolders divided by classes.
    :param embedding: Pretrained pytorch model of facenet. In embedding mode.
    :return: np.arrays of the embeddings and class labels. One element in dim=0 corresponds to one image.
    """
    X = []
    y = []
    for label, identity_path in enumerate(paths):
        print(label, identity_path)
        images = glob.glob(os.path.join(identity_path, '*.jpg')); images.sort()
        for img_path in images:
            '''
            img = Image.open(img_path)
            img = torch.FloatTensor(np.array(img).transpose((2, 0, 1)).reshape((1, 3, 160, 160)))
            '''


            img_embedding = embedding(img)
            X.append(img_embedding.detach().numpy())
            y.append(label)


    return np.array(X), np.array(y)

# Get paths of images
train_paths = glob.glob(os.path.join('train_test_split', 'train', '*'), recursive=True); train_paths.sort()
print(train_paths)
valid_paths = glob.glob(os.path.join('train_test_split', 'valid', '*'), recursive=True); valid_paths.sort()
print(valid_paths)

# Load pretrained facenet model (embedding mode)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Create embeddings
X_train, y_train = read_into_array(train_paths, resnet)
print(X_train.shape, y_train.shape)
X_valid, y_valid = read_into_array(valid_paths, resnet)
print(X_valid.shape, y_valid.shape)

# Save embeddings
save_path = 'facenet_embeddings'
np.save(os.path.join(save_path, 'X_train.npy'), X_train)
np.save(os.path.join(save_path, 'y_train.npy'), y_train)
np.save(os.path.join(save_path, 'X_valid.npy'), X_valid)
np.save(os.path.join(save_path, 'y_valid.npy'), y_valid)



