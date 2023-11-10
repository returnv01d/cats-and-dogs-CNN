import os
import shutil
import random
import wget
import constants

def copy_images_to_dir(images_to_copy, destination):
    for image in images_to_copy:
        shutil.copyfile(f'kaggle/train/{image}', f'{destination}/{image}')


def distribute_train_validation_split(validation_size_ratio=0.2):
    print("loading images form kaggle...")
    all_images = os.listdir('kaggle/train/')
    print("shuffling images...")
    random.shuffle(all_images)

    all_dogs = list(filter(lambda image: 'dog' in image, all_images))
    all_cats = list(filter(lambda image: 'cat' in image, all_images))
    print( f"dogs dataset size: {len(all_dogs)} - cats dataset size: {len(all_cats)}")

    validation_dataset_size = len(all_dogs) * validation_size_ratio
    index_to_split = int(len(all_dogs) - validation_dataset_size)
    print(f"splitting images: train dataset size: {index_to_split} - validation dataset size: {int(validation_dataset_size)}")
    training_dogs = all_dogs[:index_to_split]
    validation_dogs = all_dogs[index_to_split:]
    training_cats = all_cats[:index_to_split]
    validation_cats = all_cats[index_to_split:]

    if os.path.exists(constants.MODEL_TRAIN_DATA_DIR):
        print("deleting old model input folder")
        shutil.rmtree(constants.MODEL_TRAIN_DATA_DIR)
    else:
        print("creating model input folder")
        os.makedirs(constants.MODEL_TRAIN_DATA_DIR)

    print("making dirs for validation and train datasets...")
    os.makedirs(f'{constants.MODEL_TRAIN_DATA_DIR}train/dogs/', exist_ok=True)
    os.makedirs(f'{constants.MODEL_TRAIN_DATA_DIR}train/cats/', exist_ok=True)
    os.makedirs(f'{constants.MODEL_TRAIN_DATA_DIR}validation/dogs/', exist_ok=True)
    os.makedirs(f'{constants.MODEL_TRAIN_DATA_DIR}validation/cats/', exist_ok=True)

    print("copying and splitting data...")
    print("copying dogs train dataset")
    copy_images_to_dir(training_dogs, f'{constants.MODEL_TRAIN_DATA_DIR}train/dogs')
    print("copying dogs validation dataset")
    copy_images_to_dir(validation_dogs, f'{constants.MODEL_TRAIN_DATA_DIR}validation/dogs')
    print("copying cats train dataset")
    copy_images_to_dir(training_cats, f'{constants.MODEL_TRAIN_DATA_DIR}train/cats')
    print("copying cats validation dataset")
    copy_images_to_dir(validation_cats, f'{constants.MODEL_TRAIN_DATA_DIR}validation/cats')

def download_dataset():
    wget.download(constants.KAGGLE_DATASET_URL, out=f'{os.getcwd()}\\Data\\cats_and_dogs_filtered.zip')

download_dataset()
distribute_train_validation_split(0.2)