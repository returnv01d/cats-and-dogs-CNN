import numpy as np
import os
import tensorflow as tf
import cv2

import constants

IMG_FOLDER = r'kaggle/test/'

print("Loading model from disk...")
model = tf.keras.models.load_model('saved_model/cats_and_dogs_classifier')

for file in os.listdir(IMG_FOLDER):
    image_path = os.path.join(IMG_FOLDER, file)
    print(f"Loading {image_path} ...")

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=constants.SHAPE)

    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    prediction = model.predict(images, batch_size=10)

    print(prediction[0])

    if prediction[0] > 0:
        print(file + " is a dog")
        label = "dog"
    else:
        print(file + " is a cat")
        label = "cat"

    original = cv2.imread(image_path)

    cv2.putText(original, "Label: {}".format(label), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('', original)
    cv2.waitKey(0)

