import os
from datetime import datetime
import tensorflow as tf
import shutil


# Make dir and callback for tensorboard logging data.
def tensorboard_logging_callback(folder_prefix=''):
    log_dir = "logs/fit/inception" + folder_prefix + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"{log_dir}", exist_ok=True)

    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, embeddings_freq=1)

def copy_images_to_dir(images_to_copy, destination):
    for image in images_to_copy:
        shutil.copyfile(f'kaggle/train/{image}', f'{destination}/{image}')

def save_model(model, name):
    os.makedirs("saved_model", exist_ok=True)
    model.save(f'saved_model/{name}', overwrite=True)