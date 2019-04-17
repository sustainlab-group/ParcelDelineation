# Run python predict_model.py [model path to use] [dataframe file path] 
import sys
import os
import segmentation_models as sm
from keras.preprocessing import image
from keras.applications import resnet50, densenet, mobilenet_v2
from keras.models import Model, model_from_json
from keras.layers import Reshape, Concatenate, Conv2D, Conv2DTranspose, Dense, GlobalAveragePooling2D, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
from keras.losses import cosine_proximity
from keras import regularizers
from PIL import Image
from models.unet_dilated import unet_dilated
from segmentation_models import Unet
from random import randint
import numpy as np
import pandas as pd
import glob
import math
import warnings
import keras.backend as K
import pdb
import tensorflow as tf
from keras.models import load_model
import keras.losses
from matplotlib import pyplot as plt
from utils.metrics import get_metrics, f1, dice_coef_sim
from utils.data_loader_utils import batch_generator, batch_generator_DG, read_imgs_keraspp

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

#Set the filepaths here for laoding in the file 
is_fill = False
is_stacked = True
is_imageNet = True
is_dilated = False
image_type = 'sentinel' 


batch_size = 1
num_channels = 3
if is_stacked:
    num_channels = 9
if image_type == 'sentinel':
    input_shape = (224,224,num_channels)

base_dir = './data/' + image_type + '/'
val_file = 'parcel_segmentation_val_' + image_type if len(sys.argv) != 3 else sys.argv[2]
filepath= 'best-unet-' + image_type
csv_log_file = 'log_unet_' + image_type

sub_fill = ''
if is_fill:
    sub_fill = '_fill'

#Modify file path depending on fill/boundary task
val_file = val_file + sub_fill + '.csv'
# File path for the model
filepath = filepath + sub_fill + '.hdf5'
# Csv log file
csv_log_file = csv_log_file + sub_fill + '.csv'

#Loads validation data frame
test_df = pd.read_csv(base_dir + val_file)
pred_dir = "predictions/" + image_type + sub_fill  + '_' + str(num_channels) + '_' + str(int(is_imageNet))

if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
pred_fname = pred_dir + "unet_predictions.npy"

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

# Model file path
filepath= sys.argv[1]
pred_file= "predictions.npy" 
dependencies = {'f1':f1}

model = load_model(filepath, custom_objects=dependencies)

history = model.predict_generator(batch_generator(test_df, batch_size), steps = round(len(test_df)/batch_size))
history = history.squeeze()
np.save(pred_file, history)

predictions = np.load(pred_file) 
x_true, y_true = read_imgs_keraspp(test_df)
y_true = y_true.flatten()
y_pred = predictions.flatten()

get_metrics(y_true, y_pred, binarized=False)

print(predictions.shape)
print(predictions[0].shape)
plt.figure()
for i in range(0, 10):
  prediction = predictions[i]
  print(prediction)
  prediction[prediction > 0.5] = 255
  prediction[prediction != 255] = 0
  print(np.count_nonzero(prediction == 255))
  plt.imshow(prediction)
  plt.axis('off')
  plt.savefig(pred_dir + '/predict_unet' + str(test_df['image'][i].split('.jpeg')[0].split('_')[-1]) + '_pred.png',  bbox_inches = 'tight')

