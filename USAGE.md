# Usage of CNN
This is CNN for classyfing cats and dogs images. Taken from [tutorial](https://towardsdatascience.com/recognizing-cats-and-dogs-with-tensorflow-105eb56da35f)
This code consists of two parts - generating dataset and training it.

Tools used: python 3.6, tensorflow 2.6.2 used without GPU.
### How to set tensorflow?
`pip install tensorflow` to install packages should be enough.

### Generating dataset
The dataset is taken from [kaggle](kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition) and files are already downloaded and unzipped in kaggle folder (this is why this repo is so big).
To train a network, we need to split those images into two sets - one for training and one for validation. 
makedataset.py makis this automatically, creating appropriate datastructure for tensorflow dataloader. By default it uses 20% of images as validation set, you can change this in code variable.
It removes old structure every run and also shuffles structure, so everytime we have diferent datasets. 
Important: To make this work you need to create input_from_model dir.
Just run makedataset.py to lift off all this things up and you are done :rocket:

### Training network
After running makedataset.py you can run train.py, but be careful - close the browser and other big programs as it needs RAM.
It can also be running for a long time as we train it 100x times on our data.
When you run the script, there will be some warnings and errors about CUDA files, and some GPU sensor summary info, but those aren't important for us as we train on CPU.
After training, there should be new data in logs/fit/ folder. Those are tensorboard metrics so we can have nice visualisations. 
To see them after training run in terminal `tensorboard --logdir logs/fit --host 127.0.0.1` and click the link in command output.

