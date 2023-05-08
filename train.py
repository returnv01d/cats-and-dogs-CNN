from datetime import datetime
import os
import math
import tensorflow as tf

# Use CPU instead of GPU (GPU needs CUDA)
with tf.device('/cpu:0'):
    # Load all images and rescale them to one size
    train_imagedatagenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)
    validation_imagedatagenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)

    # Make data as generator and split it to batches
    train_iterator = train_imagedatagenerator.flow_from_directory(
        './input_for_model/train',
        target_size=(150, 150),
        batch_size=200,
        class_mode='binary')

    validation_iterator = validation_imagedatagenerator.flow_from_directory(
        './input_for_model/validation',
        target_size=(150, 150),
        batch_size=50,
        class_mode='binary')

    model = tf.keras.Sequential([
        # Conv2D + MaxPool2D - convolutional layer
        # We split the model into three major parts. First, there are three combinations of the Conv2D and MaxPool2D layers.
        # These are called convolution layers. A Conv2D layer applies a filter to the original image to amplify certain features of the picture.
        # The MaxPool2D layer reduces the size of the image and reduces the number of needed parameters needed.
        # Reducing the size of the image will increase the speed of training the network.
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPool2D((2, 2)),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Make dir and callback for tensorboard logging data.
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"{log_dir}", exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Training network.
    history = model.fit(train_iterator,
                        validation_data=validation_iterator,
                        steps_per_epoch=100,
                        epochs=5,
                        validation_steps=100,
                        callbacks=[tensorboard_callback])
