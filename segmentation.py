from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization,
                                     UpSampling2D, Conv2DTranspose, concatenate)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from os.path import join

import tensorflow.keras.backend as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import math

DATA_PATH = './dataset'
MODEL_NAME = 'segmentation_model.hdf5'
IMAGE_SIZE = (512, 512, 3)
RANDOM_SEED = 13
BATCH_SIZE = 8


def visualize(X, gt=None, prediction=True, model=None,
              apply_mask=False, output_range=10, color='DeepPink'):

    def off_ticks(axes):
        axes.set_xticks([])
        axes.set_yticks([])

    gridsize = (1, 4)
    colour = {
        'DeepPink': np.array([255, 20, 147]) / 255.,
        'LightSeaGreen': np.array([32, 178, 170]) / 255.,
        'Yellow': np.array([255, 255, 0]) / 255.,
        'OrangeRed': np.array([255, 69, 0]) / 255.,
        'Lime': np.array([0, 255, 0]) / 255.
        }

    for i in range(min(output_range, X.shape[0])):
        img =  X[i]
        if X[i].min() < 0:
            img = (img + 1) / 2
        fig = plt.figure(figsize=(15, 4))
        ax1 = plt.subplot2grid(gridsize, (0, 0))
        ax1.imshow(img)
        ax1.set_title('Original image')
        off_ticks(ax1)
        if gt is not None:
            ax2 = plt.subplot2grid(gridsize, (0, 1))
            ax2.imshow(np.squeeze(gt[i]), cmap='gray')
            ax2.set_title('Mask')
            off_ticks(ax2)
        if prediction and model is not None:
            pred = model.predict(X[i][np.newaxis, ...])[0, ...]
            pred = (pred > 0.5).astype(np.uint8)
            ax3 = plt.subplot2grid(gridsize, (0, 2))
            ax3.imshow(np.squeeze(pred), cmap='gray')
            ax3.set_title('Prediction')
            off_ticks(ax3)
            if apply_mask:
                ax4 = plt.subplot2grid(gridsize, (0, 3))
                ax4.imshow(np.clip(np.where(pred, img + colour[color], img), 0, 1))
                ax4.set_title('Apply mask')
                off_ticks(ax4)
        plt.show()
        plt.close()


def iou(y_true, y_pred):
    p = tf.cast((y_pred > 0.5), dtype='float32')
    return tf.mean(tf.sum(tf.clip(y_true * p, 0, 1), axis=(1, 2, 3)) /
                   (tf.sum(tf.clip(y_true + p, 0, 1), axis=(1, 2, 3))))


def get_encoder_decoder_model(resnet_weights='imagenet',
                              skip_connection=True, encoder_trainable=True):
    resnet_model = ResNet50(include_top=False, weights=resnet_weights,
                            input_shape=IMAGE_SIZE)
    # ENCODER
    encoder = Model(inputs=resnet_model.input,
                    outputs=resnet_model.layers[160].output,
                    name="encoder")

    # DECODER
    deconv1 = UpSampling2D(size=(2, 2))(encoder.output)
    if skip_connection:
        deconv1 = concatenate([encoder.layers[84].output, deconv1], axis=3)
    deconv1 = ZeroPadding2D(padding=(1, 1))(deconv1)
    deconv1 = Conv2D(filters=256, kernel_size=(3, 3), activation="relu",
                     kernel_initializer="he_normal")(deconv1)
    deconv1 = BatchNormalization()(deconv1)

    deconv2 = UpSampling2D(size=(2, 2))(deconv1)
    if skip_connection:
        deconv2 = concatenate([encoder.layers[38].output, deconv2], axis=3)
    deconv2 = ZeroPadding2D(padding=(1, 1))(deconv2)
    deconv2 = Conv2D(filters=128, kernel_size=(3, 3), activation="relu",
                     kernel_initializer="he_normal")(deconv2)
    deconv2 = BatchNormalization()(deconv2)

    deconv3 = UpSampling2D(size=(2, 2))(deconv2)
    if skip_connection:
        deconv3 = concatenate([encoder.layers[15].output, deconv3], axis=3)
    deconv3 = ZeroPadding2D(padding=(1, 1))(deconv3)
    deconv3 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu",
                     kernel_initializer="he_normal")(deconv3)
    deconv3 = BatchNormalization()(deconv3)

    deconv4 = UpSampling2D(size=(2, 2))(deconv3)
    if skip_connection:
        deconv4 = concatenate([encoder.layers[2].output, deconv4], axis=3)
    deconv4 = ZeroPadding2D(padding=(1, 1))(deconv4)
    deconv4 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu",
                     kernel_initializer="he_normal")(deconv4)
    deconv4 = BatchNormalization()(deconv4)

    deconv5 = UpSampling2D(size=(2, 2))(deconv4)
    deconv5 = ZeroPadding2D(padding=(1, 1))(deconv5)
    deconv5 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu",
                     kernel_initializer="he_normal")(deconv5)
    deconv5 = BatchNormalization()(deconv5)

    output_layer = Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid",
                          padding='same')(deconv5)

    encoder_decoder = Model(inputs=encoder.input,
                            outputs=output_layer,
                            name='encoder_decoder')
    if not encoder_trainable:
        for i, layer in enumerate(encoder_decoder.layers[:161]):
            layer.trainable = False

    return encoder_decoder


def get_generators_from_directory(dirname=DATA_PATH):
    train_datagen_args = dict(
        rescale=1/255.,
        rotation_range=35,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        fill_mode='constant',
        cval=0.0,
        horizontal_flip=True,
        # preprocessing_function=preprocess_input,
        dtype=np.float32
    )
    valid_datagen_args = dict(
        rescale=1/255.,
        # preprocessing_function=preprocess_input,
        dtype=np.float32
    )
    flow_args = dict(
        target_size=IMAGE_SIZE[:2],
        class_mode=None,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED
    )

    train_img_datagen = ImageDataGenerator(**train_datagen_args)
    train_img_datagen = train_img_datagen.flow_from_directory(
        directory=join(dirname, 'train/images'),
        color_mode='rgb',
        **flow_args
    )
    train_gt_datagen = ImageDataGenerator(**train_datagen_args)
    train_gt_datagen = train_gt_datagen.flow_from_directory(
        directory=join(dirname, 'train/gt'),
        color_mode='grayscale',
        **flow_args
    )
    train_generator = zip(train_img_datagen, train_gt_datagen)

    valid_img_datagen = ImageDataGenerator(**valid_datagen_args)
    valid_img_datagen = valid_img_datagen.flow_from_directory(
        directory=join(dirname, 'valid/images'),
        color_mode='rgb',
        **flow_args
    )
    valid_gt_datagen = ImageDataGenerator(**valid_datagen_args)
    valid_gt_datagen = valid_gt_datagen.flow_from_directory(
        directory=join(dirname, 'valid/gt'),
        color_mode='grayscale',
        **flow_args
    )
    valid_generator = zip(valid_img_datagen, valid_gt_datagen)

    return train_generator, valid_generator


def train_model(train_data_path):
    model = get_encoder_decoder_model()
    model.compile(optimizer=Adam(lr=1e-5),
                  loss='binary_crossentropy',
                  metrics=[iou])
    train_generator, valid_generator = get_generators_from_directory()
    # checkpointer = ModelCheckpoint(filepath=MODEL_NAME, monitor='val_iou',
    #                                verbose=1, save_best_only=True,
    #                                save_weights_only=True, mode='max')
    # logger = CSVLogger(LOG_NAME)
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=100,
        epochs=10,
        verbose=1,
        validation_data=valid_generator,
        validation_steps=10,
        # callbacks=[logger, checkpointer],
        shuffle=True,
        initial_epoch=0
    )

    return model


def predict(model, img_path):
    img = img_to_array(load_img(path=img_path), dtype=np.float32)
    img = img / 255.  # preprocess_input(img)
    ud_border = (IMAGE_SIZE[0] - img.shape[0]) / 2
    lr_border = (IMAGE_SIZE[1] - img.shape[1]) / 2
    img = np.pad(img, ((math.floor(ud_border), math.ceil(ud_border)),
                       (math.floor(lr_border), math.ceil(lr_border)), (0, 0)),
                 mode='constant')
    pred = model.predict(img[np.newaxis, ...])[0, ..., 0]
    # visualize(img[np.newaxis, ...], model=model)
    return pred[math.floor(ud_border): -math.ceil(ud_border),
                math.floor(lr_border): -math.ceil(lr_border)]
