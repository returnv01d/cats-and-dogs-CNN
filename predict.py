import os
import constants
import tensorflow as tf
import numpy as np
import cv2
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import preprocess_input, decode_predictions


IMG_FOLDER = r'kaggle/test/'

print("Loading model from disk...")
model = tf.keras.models.load_model('saved_model/cats_and_dogs_classifier')
#
# for file in os.listdir(IMG_FOLDER):
#     image_path = os.path.join(IMG_FOLDER, file)
#     print(f"Loading {image_path} ...")
#     original = cv2.imread(image_path)
#
#     image = tf.keras.utils.load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
#     image = tf.keras.utils.img_to_array(image)
#
#     image = np.expand_dims(image, axis=0)
#     image = preprocess_input(image)
#
#     print("[INFO] classifying image...")
#     predictions = model.predict(image)  # Classify the image (NumPy array with 1000 entries)
#     print(predictions)
#     P = decode_predictions(predictions)  # Get the ImageNet Unique ID of the label, along with human-readable label
#     print(P)
#
#     # Loop over the predictions and display the rank-5 (5 epochs) predictions + probabilities to our terminal
#     for (i, (imagenetID, label, prob)) in enumerate(P[0]):
#         print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
#
#     original = cv2.imread(file)
#     (imagenetID, label, prob) = P[0][0]
#     cv2.putText(original, "Label: {}, {:.2f}%".format(label, prob * 100), (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#     cv2.imshow(original)
#     cv2.waitKey(0)

im2=cv2.imread('kaggle/test/2.jpg')
im2=cv2.resize(im2, constants.SHAPE)
print(im2.shape)
img2 = tf.expand_dims(im2, 0) # expand the dims means change shape from (180, 180, 3) to (1, 180, 180, 3)
print(img2.shape)

predictions = model.predict(img2)
print(predictions)
score = tf.nn.softmax(predictions[0]) # # get softmax for each output
print(score)
class_names = ["cat", "dog"]

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)