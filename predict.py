import tensorflow as tf
import numpy as np
import os
import random
import constants
import utils

from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.saliency import Saliency

from matplotlib import pyplot as plt
from matplotlib import cm

print("Loading model from disk...")
model = tf.keras.models.load_model('saved_model/cat_and_dog_classifier')
model_copy = tf.keras.models.load_model('saved_model/cat_and_dog_classifier')


gradcam = GradcamPlusPlus(model_copy, model_modifier=ReplaceToLinear(), clone=False)
scorecam = Scorecam(model_copy, model_modifier=ReplaceToLinear(), clone=False)
saliency = Saliency(model_copy, model_modifier=ReplaceToLinear(), clone=False)


files = os.listdir(constants.IMG_FOLDER)
random.shuffle(files)
for file in files:
    img_path = os.path.join(constants.IMG_FOLDER, file)
    X = utils.load_and_preprocess_image(img_path)

    raw_predictions = model.predict(X)
    predicted_category = np.argmax(raw_predictions[0], axis=-1)

    score = CategoricalScore([predicted_category])
    print(raw_predictions)

    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))

    prediction_info = "Predicted: {} ({})".format(constants.CLASS_INFO[predicted_category], utils.make_prediction_labels(raw_predictions))
    ax[0].set_title(prediction_info, fontsize=10)
    ax[0].imshow(X[0])
    ax[0].axis("off")

    ax[1].set_title("GradCAM++", fontsize=16)
    cam = gradcam(score, X, penultimate_layer=-1)
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    ax[1].imshow(X[0])
    ax[1].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[1].axis('off')

    ax[2].set_title("ScoreCAM++", fontsize=16)
    sccam = scorecam(score, X, penultimate_layer=-1)
    heatmap = np.uint8(cm.jet(sccam[0])[..., :3] * 255)
    ax[2].imshow(X[0])
    ax[2].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[2].axis('off')

    ax[3].set_title("Saliency map", fontsize=16)
    saliency_map = saliency(score, X, smooth_samples=20, smooth_noise=0.20)
    ax[3].imshow(saliency_map[0], cmap='jet')
    ax[3].axis('off')

    plt.tight_layout()
    plt.show()

    input()
