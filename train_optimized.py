# Code here was meant to show transfer learning, but weren't updated (and coulddn't work) because
# tf-kersa-vis visualisations doesn't work with transfer learning or Inception model.

import os
from matplotlib import pyplot as plt
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
    batch_size=constants.BATCH_SIZE,
    class_mode='categorical')

validation_iterator = validation_imagedatagenerator.flow_from_directory(
    './input_for_model/validation',
    target_size=constants.SHAPE,
    batch_size=constants.BATCH_SIZE,
    class_mode='categorical',
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
x = layers.Dropout(0.2)(x)
x = layers.Dense(2, activation='softmax')(x)

model = Model(pre_trained_model.input, x)
model.compile(
    optimizer=RMSprop(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Use CPU instead of GPU (GPU needs CUDA)
with tf.device('/cpu:0'):
    # Training network.
    history = model.fit_generator(
        generator=train_iterator,
        validation_data=validation_iterator,
        steps_per_epoch=100,
        epochs=12,
        validation_steps=100,
        verbose=2,
        # callbacks=[utils.tensorboard_logging_callback("inception")],
    )
    acc      = history.history[     'accuracy' ]
    val_acc  = history.history[ 'val_accuracy' ]
    loss     = history.history[    'loss' ]
    val_loss = history.history['val_loss' ]

    epochs   = range(len(acc)) # Get number of epochs

    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.plot  ( epochs,     acc, label='Training Accuracy' )
    plt.plot  ( epochs, val_acc, label='Validation Accuracy' )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title ('Training and validation accuracy')
    plt.figure()

    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot  ( epochs,     loss, label='Training Loss' )           
    plt.plot  ( epochs, val_loss, label='Validation Loss' )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title ('Training and validation loss'   )
    plt.show()

utils.save_model(model, "cats_and_dogs_classifier_optimized")