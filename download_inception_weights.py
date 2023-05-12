import os
import wget

url = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
wget.download(url, out=f'{os.getcwd()}/kaggle/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

