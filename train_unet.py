from keras.preprocessing import image
import segmentation_models as sm
from segmentation_models import Unet
from keras.applications import resnet50, densenet, mobilenet_v2
from keras.models import Model, model_from_json
from keras.layers import Reshape, Concatenate, Conv2D, Conv2DTranspose, Dense, GlobalAveragePooling2D, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
from keras.losses import cosine_proximity
from keras import regularizers
from PIL import Image
from random import randint
from models.unet import unet
from models.unet_dilated import unet_dilated
from utils.data_loader_utils import batch_generator, batch_generator_DG
from utils.metrics import *
import numpy as np
import pandas as pd
import glob
import math
import warnings
import keras.backend as K
import keras
import pdb
import tensorflow as tf
import cv2

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

def learning_rate_scheduler(epoch):
    lr = 1e-4
    '''
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 150:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    '''
    print("Set Learning Rate : {}".format(lr))
    return lr


#Set the variables here for training the model 
is_fill = False
is_stacked = True
is_imageNet = True
is_dilated = False # dilated models are only for non-pretrained models 
image_type = 'sentinel' 

num_channels = 3
if is_stacked:
    num_channels = 9

input_shape = (224,224,num_channels)
batch_size = 6
base_dir = './data/' + image_type + '/'
train_file = 'parcel_segmentation_train_' + image_type
val_file = 'parcel_segmentation_val_' + image_type
filepath= 'best-unet-' + image_type
csv_log_file = 'log_unet_' + image_type

sub_fill = ''
if is_fill:
    sub_fill = '_fill'

#Modify file path depending on fill/boundary task
train_file = train_file + sub_fill + '.csv'
val_file = val_file + sub_fill + '.csv'
filepath = filepath + sub_fill + '.hdf5'
csv_log_file = csv_log_file + sub_fill + '.csv'


#Loads training and validation data frame
#Dataframe contains the paths of the images
train_df = pd.read_csv(base_dir + train_file)
val_df = pd.read_csv(base_dir + val_file)

model = None 

if is_dilated:
    model = unet_dilated(input_size = input_shape)
elif is_imageNet:
    model_unet = Unet(BACKBONE, encoder_weights='imagenet')
    if is_stacked: 
        new_model = keras.models.Sequential()
        new_model.add(Conv2D(3, (1,1), padding='same', activation='relu', input_shape=input_shape))
        new_model.add(model_unet)
        model = new_model
    else:
        model = model_unet
else:
    model = unet(input_size=input_shape)


model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=learning_rate_scheduler(0)),
              metrics=['acc', f1])

checkpoint = ModelCheckpoint(filepath, monitor='f1', verbose=1, save_best_only=True, mode='max')
csv_logger = CSVLogger(csv_log_file, append=True, separator=';')
callbacks_list = [checkpoint, csv_logger]

model.fit_generator(batch_generator(train_df, batch_size, is_stacked, is_imageNet), steps_per_epoch=round((len(train_df))/batch_size),
        epochs=200, validation_data=batch_generator(val_df, batch_size), validation_steps=round((len(val_df))/batch_size),callbacks=callbacks_list)
