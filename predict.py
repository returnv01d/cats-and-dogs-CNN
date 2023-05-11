import os
import math
import tensorflow as tf
import numpy as np
import cv2

IMG_WIDTH = 150
IMG_HEIGHT = 150
img_folder = r'C:\Users\saksh\Downloads\training_set\training_set'


def create_dataset(img_folder):
    img_data_array = []

    for file in os.listdir(os.path(img_folder,)):
        image_path = os.path.join(img_folder, file)
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        try:
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
        except:
            break
        image = np.array(image)
        image = image.astype('float32')
        image /= 255
        img_data_array.append(image)

    return img_data_array

# Use CPU instead of GPU (GPU needs CUDA)
with tf.device('/cpu:0'):
    # Load all images and rescale them to one size
    predict_imagedatagenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)
    test_generator = predict_imagedatagenerator('kaggle/',
                                  # only read images from `test` directory
                                  classes=['test'],
                                  # don't generate labels
                                  class_mode=None,
                                  # don't shuffle
                                  shuffle=False,
                                  # use same size as in training
                                  target_size=(299, 299))

    model = tf.keras.models.load_model('saved_model/cats_and_dogs_clasifier')

    preds = model.predict_generator(test_generator)
    preds_cls_idx = preds.argmax(axis=-1)

    idx_to_cls = {v: k for k, v in predict_imagedatagenerator.class_indices.items()}
    preds_cls = np.vectorize(idx_to_cls.get)(preds_cls_idx)
    filenames_to_cls = list(zip(test_generator.filenames, preds_cls))