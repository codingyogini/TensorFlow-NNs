# Building Neural Networks with Tensorflow 

### Environment

MacOS 10.13 64-bit
Python 3.6 framework 
Anaconda 4.6
Tensorflow CPU version 1.12

### Installing TensorFlow and dependencies

Instructions for installing Tensorflow can be found [here](https://www.tensorflow.org/install/)

First, initiate a virtualenv, then install tensorflow

	- installed the basic tensorflow without GPU support using pip.

	- For me, this command line didn't work - 
		pip install --upgrade tensorflow
	  but this did: python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl

	- Install matplotlib: python -m pip install -U matplotlib //requires python as a framework on MacOS

	- Install TensorFlow Probability toolbox:
		pip install --upgrade tfp-nightly

More about the TF toolbox: https://medium.com/tensorflow/introducing-tensorflow-probability-dca4c304e245

Note: edward (now edward2) has been ported to tensorflow_probability:
https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/edward2/Upgrading_From_Edward_To_Edward2.md

### Neural Network - classify fashion images 

##### Dependencies:
- tensorflow
- tf.keras 
- numpy
- matplotlib

##### Dataset:
- the fasion MNIST dataset: https://github.com/zalandoresearch/fashion-mnist
- contains 70,000 greyscale images in 10 categories 
- the images show individual articles of clothing at low resolution (28x28px)
- Training set: 60,000 images, Testing set: 10,000 images

##### Model: 
- Keras Sequential model (linear stack of layers)
- 3 layers with activation functions: rectified linear and softmax
- loss function: sparse_categorical_crossentropy
- optimizer: adam
- metrics: accuracy 
    
##### Outcome:
- Accuracy: ~87% on training data, ~87% on test data

##### Code:
	NN_MNIST.pynb

### Bayesian Neural Network - classify handwritten digits

##### Dependencies:
- tensorflow
- tf.keras 
- numpy
- matplotlib
- pandas
- seaborn
- edward

##### Dataset:
- the MNIST dataset of handwritten digits: http://yann.lecun.com/exdb/mnist/
- Training set: 60,000 images, Testing set: 10,000 images
- 10 categories: {0,1,..9}
- 28x28 image = 784 features  
- nonMNIST images can be found here: http://yaroslavvb.com/upload/notMNIST/

##### Model:
	Task: classify the handwritten MNIST digits into one of the 10 classes {0,1,2,...9} and give a measure of the uncertainty of the classificatin. 
	Model: soft-max regression 
	Likelihood function: Categorical likelihood function - to quantify the probability of the observed data given a set of parameters, weights and biases in this case. The Categorical distrubution is also known as the Multinoulli distribution. 

	Infer the posterior using Variational Inference - minimize the KL divergence between the the true posterior and approximating distributions 

	We evaluate the model with a set of predictions and their accuracies, instead of a single prediction, as per regular NNs.

##### Code:
	BayesianNN_MNIST.pynb

