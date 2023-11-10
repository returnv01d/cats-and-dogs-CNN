import matplotlib.pyplot as plt
import tensorflow as tf

import constants
import utils

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

# Flow training images in batches of 20 using train_datagen generator
train_ds = train_datagen.flow_from_directory(
    constants.MODEL_TRAIN_DATA_DIR + "train/",
    batch_size=constants.BATCH_SIZE,
    class_mode='categorical',
    target_size=constants.SHAPE,
)

# Flow validation images in batches of 20 using test_datagen generator
val_ds = test_datagen.flow_from_directory(
    constants.MODEL_TRAIN_DATA_DIR + "validation/",
    batch_size=constants.BATCH_SIZE,
    class_mode='categorical',
    target_size=constants.SHAPE,
)

class_names = train_ds.class_indices
print(class_names)

num_classes = len(class_names)

model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),

    layers.Dense(num_classes, name="outputs", activation='softmax'),
])

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.build(input_shape=(None, 256, 256, 3))
model.summary()

epochs = 12
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

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
plt.show()

utils.save_model(model, "cat_and_dog_classifier")
