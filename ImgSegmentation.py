from fnmatch import fnmatchcase
import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
from IPython.core.display import display_pretty
from IPython.display import Image, display

import tensorflow as tf
from keras.preprocessing.image import load_img

import PIL
from PIL import ImageOps
from tensorflow import keras
from tensorflow.keras import layers
import random, copy

from tensorflow.keras import layers
from tensorflow.keras.models import load_model


import numpy as np
import pandas as pd
import tensorflow as tf
import time,os, math

# GPU Setting
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:   
    print(e)


# 전역변수
img_size = (128, 128)
## Hyperparam
epochs = 7000
num_classes = 3
batch_size = 64
val_samples = 4

class DataPIH(keras.utils.Sequence):
  def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
    self.batch_size = batch_size
    self.img_size = img_size
    self.input_img_paths = input_img_paths
    self.target_img_paths = target_img_paths

  def __len__(self):
    return len(self.target_img_paths) // self.batch_size

  def __getitem__(self, idx):
    i = idx * self.batch_size
    batch_input_img_paths = self.input_img_paths[i:i+self.batch_size]
    batch_target_img_paths = self.target_img_paths[i:i+self.batch_size]

    x = np.zeros((self.batch_size, ) + self.img_size + (3, ), dtype="float32")
    for j, path in enumerate(batch_input_img_paths):
      img = load_img(path, target_size=self.img_size)
      x[j] = img

    y = np.zeros((self.batch_size, ) + self.img_size + (1, ), dtype="uint8")
    for j, path in enumerate(batch_target_img_paths):
      img = load_img(path, target_size=self.img_size, color_mode="grayscale")
      y[j] = np.expand_dims(img, 2)
      y[j] -= 1
    
    return x, y

def train_model(input_dir, target_dir, weight_save_path):
  try:
      with tf.device('/device:GPU:0'):

        # 이미지 경로를 가져와서 붙인 후 리스트로 가지고 있음
        input_img_paths = sorted([input_dir+"/"+fname
                        for fname in os.listdir(input_dir)
                        if fname.endswith(".jpg") ])

        # 마스크 이미지 경로를 가져와서 리스트로 변환
        target_img_paths = sorted([target_dir+"/"+fname
                        for fname in os.listdir(target_dir)
                        if fname.endswith(".png") and not fname.startswith(".") ])

        ## 데이터전처리
        random.Random(1337).shuffle(input_img_paths)
        random.Random(1337).shuffle(target_img_paths)

        train_input_img_paths = input_img_paths[:-val_samples]
        train_target_img_paths = target_img_paths[:-val_samples]
        val_input_img_paths = input_img_paths[-val_samples:]
        val_target_img_paths = target_img_paths[-val_samples:]

        train_gen = DataPIH(batch_size, img_size, train_input_img_paths, train_target_img_paths)
        val_gen = DataPIH(batch_size, img_size, val_input_img_paths, val_target_img_paths)


        ## 모델구성
        model = get_model(img_size, num_classes)
        model.summary()
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics = ["accuracy"])#  sparse_categorical_crossentropy

        callbacks = [keras.callbacks.ModelCheckpoint("checkpoint/checkpointpview_segmentation.h5", save_best_only=True)]

        print(len(train_gen.input_img_paths))
        print(len(train_gen.target_img_paths))
        print(len(val_gen.input_img_paths))
        print(len(val_gen.target_img_paths))

        ## 모델학습
        model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

        model.save(weight_save_path)
        
  except RuntimeError as e:
      pass
  

def get_model(img_size, num_classes):
  inputs = keras.Input(shape=img_size + (3, ))
  
  x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  previous_block_activation = x

  for filters in [64, 128, 256]:
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    residual = layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
    x = layers.add([x, residual])
    previous_block_activation = x
  
  for filters in [256, 128, 64, 32]:
    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(2)(x)

    residual = layers.UpSampling2D(2)(previous_block_activation)
    residual = layers.Conv2D(filters, 1, padding="same")(residual)
    x = layers.add([x, residual])
    previous_block_activation = x

  outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

  model = keras.Model(inputs, outputs)
  return model


def display_mask(val_preds, i):
  mask = np.argmax(val_preds[i], axis=-1)
  mask = np.expand_dims(mask, axis=-1)
  img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
  img = keras.preprocessing.image.array_to_img(mask)
  img = np.array(img)

  return img  

def test_model(img_path, model_path):
  img = cv2.imread(img_path)

  cv2.imshow("origin_img", img)
  height, width, _ = img.shape
  img = cv2.resize(img, img_size)
  img = tf.expand_dims(img, axis= 0)
  model = load_model(model_path)
  model.summary()
  tf.keras.utils.plot_model(model, show_shapes=True)

  ## 추론 2.
  preds = model.predict(img)
  img2 = display_mask(preds, 0)
  

  cv2.imshow("result_img", cv2.resize(img2, (width, height)))
  cv2.waitKey(0)


if __name__ == "__main__":
  input_dir = "data/img" # 입력데이터 디렉터리 경로
  target_dir = "data/preprocessed_mask" # 마스크 데이터 디렉터리 경로
  train_model(input_dir, target_dir, "weight_231023_1.h5")

  # test_model("data/img/Abyssinian_1.jpg", "weight_231018_1.h5")