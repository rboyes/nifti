import nibabel as nib

import datetime
import time
import math
import nibabel as nib
import pandas as pd
import os
import random
import sys
import numpy as np

if (len(sys.argv) < 2):
    print('Usage: NiftiSegmenter.py train trainingFiles.csv validationFiles.csv model.h5 log.txt')
    print('       Or')
    print('Usage: NiftiSegmenter.py test model.h5 input.nii segmentation.nii')
    sys.exit(1)


import tensorflow.keras.backend as keras_backend
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPool3D, UpSampling3D
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam


def read_niftis(image_paths, mask_paths):

    if len(image_paths) != len(mask_paths):
        raise ValueError('The input files and segmentation files are different sizes')

    if len(image_paths) == 0:
        raise ValueError('Cannot have zero length')

    num_images = len(image_paths)
    for index in range(num_images):
        image_path = image_paths[index]
        mask_path = mask_paths[index]
        if not os.path.isfile(image_path):
            raise ValueError('file %s does not exist ' % image_path)
        if not os.path.isfile(mask_path):
            raise ValueError('file %s does not exist ' % mask_path)

    input_image = nib.load(image_paths[0])
    (nx, ny, nz) = input_image.shape[0:3]
    input_images = np.ndarray((num_images, nx, ny, nz, 1), dtype=np.float32)
    mask_images = np.ndarray((num_images, nx, ny, nz, 1), dtype=np.float32)
    for index in range(0, num_images):
        input_image = nib.load(image_paths[index])
        mask_image = nib.load(mask_paths[index])
        input_images[index, :, :,:, 0] = input_image.get_fdata(dtype=np.float32)
        mask_images[index, :, :, :, 0] = mask_image.get_fdata(dtype=np.float32)
    return input_images, mask_images


def read_csv(csv_path, shuffle=False):

    df_filenames = pd.read_csv(csv_path, header=None)
    input_names = df_filenames[0].values.tolist()
    output_names = df_filenames[1].values.tolist()

    if shuffle:
        combined = list(zip(input_names, output_names))
        random.shuffle(combined)
        input_names, output_names = zip(*combined)

    return input_names, output_names


def overlap(ytrue, ypred):
    ytrue_flat = keras_backend.flatten(ytrue)
    ypred_flat = keras_backend.flatten(ypred)
    intersection = keras_backend.minimum(ytrue_flat, ypred_flat)
    union = keras_backend.maximum(ytrue_flat, ypred_flat)
    overlap = keras_backend.sum(intersection) / keras_backend.sum(union)
    return overlap


def probabilistic_overlap(ytrue, ypred):

    ytrue_flat = keras_backend.flatten(ytrue)
    ypred_flat = keras_backend.flatten(ypred)
    intersection = ytrue_flat * ypred_flat
    union = (ytrue_flat + ypred_flat) - intersection
    overlap = keras_backend.sum(intersection) / keras_backend.sum(union)
    return overlap


def overlap_loss(ytrue, ypred):
    return 1.0 - overlap(ytrue, ypred)


def create_unet(nx, ny, nz):

    inputs = Input((nx, ny, nz, 1))
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPool3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)

    up5 = UpSampling3D(size=(2, 2, 2))(conv4)
    merge5 = concatenate([up5, conv3])
    conv5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(merge5)
    conv5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = UpSampling3D(size=(2, 2, 2))(conv5)
    merge6 = concatenate([up6, conv2])
    conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(merge6)
    conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = UpSampling3D(size=(2, 2, 2))(conv6)
    merge7 = concatenate([up7, conv1])
    conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(merge7)
    conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv7)

    conv8 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=conv8)

    model.compile(optimizer=Adam(lr=1.0E-5), loss=overlap_loss)

    return model


if "train" in sys.argv[1]:

    training_paths = sys.argv[2]
    validation_paths = sys.argv[3]
    model_path = sys.argv[4]
    log_path = sys.argv[5]
    (training_image_paths, training_mask_paths) = read_csv(training_paths, shuffle=True)
    (validation_image_paths, validation_mask_paths) = read_csv(validation_paths)

    (validation_images, validation_masks) = read_niftis(validation_image_paths, validation_mask_paths)

    model = create_unet(validation_images.shape[1], validation_images.shape[2], validation_images.shape[3])

    logger = CSVLogger(log_path, append=True)
    checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)

    chunk_size = 64
    num_epochs = 80
    num_images = len(training_image_paths)
    for epoch in range(num_epochs):
        for index in range(0, num_images, chunk_size):
            end_index = min(index + chunk_size, num_images)
            image_paths = training_image_paths[index:end_index]
            mask_paths = training_mask_paths[index:end_index]
            (training_images, training_masks) = read_niftis(image_paths, mask_paths)
            model.fit(training_images, training_masks, epochs=1, batch_size=1, verbose=1, shuffle=True,
                      validation_data=(validation_images, validation_masks), callbacks=[checkpointer, logger])

elif "test" in sys.argv[1]:
    model_path = sys.argv[2]
    image_path = sys.argv[3]
    output_path = sys.argv[4]

    model = load_model(model_path, custom_objects={'overlapLoss': overlap_loss, 'overlap': overlap})
    nifti = nib.load(image_path)
    (nx, ny, nz) = nifti.shape[0:3]
    image = np.ndarray((1, nx, ny, nz, 1), dtype=np.float32)
    image[0, :, :, :, 0] = nifti.get_fdata(dtype=np.float32)

    prediction = model.predict(image, batch_size=1)
    nifti_prediction = nib.Nifti1Image(prediction[0, :, :, :, 0], affine=nifti.affine, header=nifti.header)
    nib.save(nifti_prediction, output_path)
