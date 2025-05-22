## import the necessary packages
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense

"""
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from shutil import copy
from random import random
from random import seed
from matplotlib import pyplot
from glob import glob
import cv2
from os import path, environ, listdir, makedirs
import numpy as np
import matplotlib.pyplot as plt
# Fix: tensorflow.python.framework.errors_impl.InternalError: Blas GEMM launch failed : a.shape=(64, 67), b.shape=(67, 67), m=64, n=67, k=67
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.compat.v1.Session(config=config)

environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 64
drive_dir = ""
load_path = drive_dir + "models/" + "facenet_keras.h5"
save_path = drive_dir + "models/"


def define_model(drop, load_path):
    model = Sequential()

    """
    model.add(Conv2D(batch_size,(3,3), padding='same', input_shape=(224,224,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(batch_size,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(drop))


    model.add(Conv2D(128,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(drop))

    model.add(Conv2D(256,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(drop))
    model.add(Conv2D(256,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(drop))

    model.add(Conv2D(512,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(drop))
    model.add(Conv2D(512,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(drop))

    model.add(Conv2D(512,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(drop))
    model.add(Conv2D(512,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(drop))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    """

    """
    base_model = tf.keras.models.load_model(load_path)
    base_model.trainable = False

    #inputs = keras.Input(shape=(224, 224, 3))
    #x = base_model(inputs, training=False)

    dense = Sequential()
    dense.add(Dense(67))
    dense.add(Activation('relu'))
    dense.add(Dense(67))
    dense.add(Activation('softmax'))
    #outputs = dense(x)
    outputs = dense(base_model)

    model = keras.Model(outputs)
    """

    #model.add(Dense(67))
    #model.add(Activation('relu'))
    #model.add(Dense(67))
    #model.add(Activation('softmax'))

    base_model = keras.applications.VGG19(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        include_top=True,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,)  # Do not include the ImageNet classifier at the top.

    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    x = base_model(inputs)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.Dense(67)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dense(67)(x)
    x = keras.layers.Activation('softmax')(x)

    # A Dense classifier with a single unit (binary classification)
    outputs = x
    model = keras.Model(inputs, outputs)

    #opt = keras.optimizers.Adam()
    opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.1)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['mae', 'accuracy'])

    print('Created Model')
    return model


def split_data(dataDir):
    subdirs = ['train_test_split/train/', 'train_test_split/valid/']
    for subdir in subdirs:
        for labeldir in glob(dataDir + '/*'):
            newdir = subdir + labeldir.split('/')[1]
            try:
                makedirs(newdir)
            except:
                print('Already Created')
    seed(1)
    val_ratio = 0.2
    for file in listdir(dataDir):
        for images in listdir(dataDir + '/' + file):
            src = dataDir + '/' + file + '/' + images
            #            print(src)
            dst_dir = 'train_test_split/train/'
            ran = random()
            if ran < val_ratio:
                dst_dir = 'train_test_split/valid/'
            if file.startswith(file):
                dst = dst_dir + file
                print(dst)
                copy(src, dst)


def disp_data(dataDir):
    for cl in glob(dataDir + '/*'):
        for i, ids in enumerate(glob(cl + '/*')):
            pyplot.subplot(330 + 1 + i)
            im = cv2.imread(ids)
            img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, dtype=np.uint8)
            pyplot.imshow(img)
            if i == 8:
                break
        pyplot.show()


def display_results(history, ep, d):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(ep)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('finTest/results{:d}_{:.1f}.png'.format(ep, d))
    plt.show()


mods = []
drops = [0.4]
epochs = [20]
for d in drops:
    for ep in epochs:
        if not path.exists(drive_dir + 'finTest/model{:d}_{:.1f}.h5'.format(ep, d)):

            dataDir = 'aug'
            if not path.exists(drive_dir + 'train_test_split'):
                split_data(dataDir)
            else:
                print('Already Split')

            #    ep = 50
            model = define_model(d, load_path)
            print(model.summary())

            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
            val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

            train = train_datagen.flow_from_directory(
                'train_test_split/train/', shuffle=True,
                class_mode='categorical', batch_size=batch_size, target_size=(224, 224))
            test = val_datagen.flow_from_directory(
                'train_test_split/valid/', shuffle=True,
                class_mode='categorical', batch_size=batch_size, target_size=(224, 224))

            history = model.fit(train, epochs=ep, validation_data=(test))
            model.save('finTest/model{:d}_{:.1f}.h5'.format(ep, d))
            display_results(history, ep, d)
        else:
            print('Already computed')

#
#_,mae, acc = model.evaluate_generator(valid, steps=len(valid), verbose=0)
#print('> %.3f' % (acc * 100.0))
#
#

#model.save("modelor.h5")
