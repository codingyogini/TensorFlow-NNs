# Tensorflow-examples

### Installing TensorFlow ++

Macbook Air HighSierra 10.13.6
Python 3.7 (Installed with Anaconda3)

Instructions for installing Tensorflow can be found [here](https://www.tensorflow.org/install/)

First, initiate a virtualenv, then install tensorflow

	- installed the basic tensorflow without GPU support using pip.

	- For me, this command line didn't work - 
		pip install --upgrade tensorflow
	  but this did: python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl

	- Install matplotlib: python -m pip install -U matplotlib //requires python as a framework on MacOS

	- Install TensorFlow Probability toolbox:
		pip install --user --upgrade tfp-nightly

More about the TF toolbox: https://medium.com/tensorflow/introducing-tensorflow-probability-dca4c304e245


### Neural Network - classify fashion images 

Followed the TensorFLow tutorial :
https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb

Dependencies:
	tensorflow
	tf.keras 
	numpy
	matplotlib

Dataset:
	- the fasion MNIST dataset: https://github.com/zalandoresearch/fashion-mnist
	- contains 70,000 greyscale images in 10 categories 
	- the images show individual articles of clothing at low resolution (28x28px)
	- Training set: 60,000 images, Testing set: 10,000 images

Model: 
    - Keras Sequential model (linear stack of layers)
    - 3 layers with activation functions: rectified linear and softmax
    - loss function: sparse_categorical_crossentropy
    - optimizer: adam
    - metrics: accuracy 
    
Outcome:
    - Accuracy: ~87% on training data, ~87% on test data

Code:
	NN_MNIST.pynb



