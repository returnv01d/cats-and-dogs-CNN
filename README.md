Code developed to make CNN used to show neural network learning and classifying process for AI workshop for students in my old vocational school.

# Usage of repo
This is code for training categorical CNN for classifying cats and dogs images, and then predicting images and showing visualizations.
Code were taken and built from many tutorials.
This code consists of three parts - generating dataset, training model, and making predictions.

Tools used: python 3.11, tensorflow 2.12 used without GPU and tf-keras-vis for prediction visualizations and matplotlib for drawing images.
### How to set packages?
`pip install tensorflow numpy matplotlib tf_keras_vis wget` to install packages should be enough.

### Generating dataset
The dataset is taken from [kaggle](kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition) and files are already downloaded and unzipped in `kaggle` folder (this is why this repo is so big).
To train a network, we need to split those images into two sets - one for training and one for validation. 
`make_dataset.py` makes this automatically, creating appropriate data structure for tensorflow dataloader. 
By default, it uses 20% of images as validation set, you can change this in code variable.
It removes old structure every run and also shuffles structure, so everytime we have different datasets.
Just run `make_dataset.py` to lift off all this things up, and you are done :rocket:

### Training network
After running `make_dataset.py` you can run `train.py`, but be careful - close the browser and other big programs as it needs RAM.
It can run for a long time! (few hours)
You can tweak up ram usage and training speed by:
- reducing dataset size (there are 25k images, you can try on lesser set but it will be less accurate)
- reducing trained images size (they are scaled to 256x256, and this is big. You can control this with `constants.py` consts)
- reducing number of epochs or validation/training steps
- stopping logging data for tensorboard during training - remove `tensor_callback` usage in `train.py`
When you run the script, there will be some warnings and errors about CUDA files, but those aren't important for us as we train on CPU.
After training, the plot with training accuracy and loss will be shown and the trained model will be saved to file.

There are two files for training - `train` and `train_optimized`.
The second file was meant to show transfer learning and optimization approaches, but the Inception model used for transfer learning doesn't work with tf-keras-vis visualizations, so the code was abandoned.

#### Checking and debugging model training process
After training, if you don't disable tensorboard_callback, there should be new data in `logs/fit/` folder. Those are tensorboard metrics, so we can have nice visualisations about training - how the accuracy parameters such as loss and validation accuracy changed.
To see them after training run in terminal `tensorboard --logdir logs/fit --host 127.0.0.1` and click the link in command output.

### Making predictions
To get images for predictions, put your images in `kaggle/train` folder and run `predict.py`. 
Script will show images with predicted labels along with visualizations, press enter in console to go to next image.
