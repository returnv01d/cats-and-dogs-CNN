import os

import constants
import utils
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model
from tensorflow.keras import layers


# Load all images and rescale them to one size
train_imagedatagenerator = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
validation_imagedatagenerator = ImageDataGenerator(rescale=1/255.0)

# Make data as generator and split it to batches
train_iterator = train_imagedatagenerator.flow_from_directory(
    './input_for_model/train',
    target_size=constants.SHAPE,
    batch_size=200,
    class_mode='binary')

validation_iterator = validation_imagedatagenerator.flow_from_directory(
    './input_for_model/validation',
    target_size=constants.SHAPE,
    batch_size=50,
    class_mode='binary',
)

local_weights_file = f'{os.getcwd()}/kaggle/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
# the network shape of Inception V3 is integrated in Keras
pre_trained_model = InceptionV3(
    input_shape=(constants.IMG_WIDTH, constants.IMG_HEIGHT, 3),
    include_top=False,
    weights=None,
)
# combine layout and weights
pre_trained_model.load_weights(local_weights_file)

# make pre-trained model read-only
for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

# set layer "mixed 7" as end of pre-trained network
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(
    optimizer=RMSprop(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

# Use CPU instead of GPU (GPU needs CUDA)
with tf.device('/cpu:0'):
    # Training network.
    model.fit(
        train_iterator,
        validation_data=validation_iterator,
        steps_per_epoch=100,
        epochs=10,
        validation_steps=50,
        callbacks=[utils.tensorboard_logging_callback("inception")],
    )

utils.save_model(model, "cats_and_dogs_classifier_optimized")