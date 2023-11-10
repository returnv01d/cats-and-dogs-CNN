import os
from datetime import datetime
import tensorflow as tf
import cv2
import constants

import numpy as np

from tensorflow.keras.preprocessing.image import load_img
from matplotlib import cm


# Make dir and callback for tensorboard logging data.
def tensorboard_logging_callback(folder_prefix=''):
    log_dir = "logs/fit/inception" + folder_prefix + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"{log_dir}", exist_ok=True)

    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, embeddings_freq=1)

def save_model(model, name):
    os.makedirs("saved_model", exist_ok=True)
    model.save(f'saved_model/{name}', overwrite=True)


def draw_text_with_bg(img, text, font=cv2.FONT_HERSHEY_PLAIN, pos=(0, 0), font_scale=1.3, font_thickness=2, text_color=(255, 255, 255), text_color_bg=(0, 0, 0)):
    x, y = pos
    alpha = 0.8
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    overlay = img.copy()
    cv2.rectangle(overlay, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(overlay, text, (x, y + text_h), font, font_scale, text_color, font_thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    return text_size

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=constants.SHAPE)
    img = np.asarray(img)
    images = np.asarray([img])
    X = images / 255

    return X

def add_prediction_info(img, raw_predictions):
    predicted_category = np.argmax(raw_predictions[0], axis=-1)
    label = "Label: {}".format(constants.CLASS_INFO[predicted_category])

    draw_text_with_bg(img, label, pos=(10, 10))
    draw_text_with_bg(img, make_prediction_labels(raw_predictions=raw_predictions), pos=(10, 30))

def superimpose_heatmap(heatmap, overlay):
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = resized_image(heatmap, overlay)
    cv2.addWeighted(heatmap, 0.4, overlay, 1 - 0.4, 0, overlay)

def resized_image(img, original):
    h, w, c = original.shape
    return cv2.resize(img, dsize=(w, h))

def show_visualizations(img_path, X, score, raw_prediction, gradcam, saliency):
    original = cv2.imread(img_path)
    overlay = original.copy()
    add_prediction_info(original, raw_prediction)
    cv2.imshow("Prediction", original)

    cam = gradcam(score, X, penultimate_layer=-1)
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    superimpose_heatmap(heatmap, overlay)
    cv2.imshow('GradCam++', overlay)

    saliency_map = saliency(score, X, smooth_samples=20, smooth_noise=0.20)
    saliency_map = resized_image(saliency_map[0], overlay)
    cv2.imshow("Saliency Map", saliency_map)


def make_prediction_labels(raw_predictions):
    prediction_labels = ["{}: {}%".format(constants.CLASS_INFO[i], round(pred * 100, 2)) for i, pred in enumerate(raw_predictions[0])]
    return ", ".join(prediction_labels)