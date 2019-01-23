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



### Bayesian Neural Network 

A Bayesian Neural Network is a neural network with a prior distribution on its weights and biases (Neil, 2012). It provides improved uncertainty about its predictions via these priors. 

Dependencies:
	tensorflow
	tensorflow_probability

Dataset:
	40 random floats

Code: 
	Bayesian_NN.py 

# Regression Model Example:

Consider a data set $\{(\mathbf{x}_n, y_n)\}$, where each data point comprises of features $\mathbf{x}_n\in\mathbb{R}^D$ and output $y_n\in\mathbb{R}$. Define the likelihood for each data point as 
$$\begin{aligned} p(y_n \mid \mathbf{w}, \mathbf{x}_n, \sigma^2) &amp;= \text{Normal}(y_n \mid \mathrm{NN}(\mathbf{x}_n\;;\;\mathbf{w}), \sigma^2),\end{aligned}$$


where $\mathrm{NN}$ is a neural network whose weights and biases form the latent variables $\mathbf{w}$. Assume $\sigma^2$ is a known variance.

Define the prior on the weights and biases $\mathbf{w}$ to be the standard normal $$\begin{aligned} p(\mathbf{w}) &amp;= \text{Normal}(\mathbf{w} \mid \mathbf{0}, \mathbf{I}).\end{aligned}$$

First, build the model in Edward, defining a 3-layer Bayesian neural network with $\tanh$ nonlinearities.

# Classification Model Example:

Consider a data set $\{(\mathbf{x}_n, y_n)\}$, where each data point comprises of features $\mathbf{x}_n\in\mathbb{R}^D$ and output $y_n\in{\{0,1}\}$. Define the likelihood for each data point as $$\begin{aligned} p(y_n \mid \mathbf{w}, \mathbf{x}_n) &amp;= \mathrm{NN}(\mathbf{x}_n\;;\;\mathbf{w})^{y_n} (1-\mathrm{NN}(\mathbf{x}_n\;;\;\mathbf{w}))^{1-y_n},\end{aligned}$$

where $\mathrm{NN}$ denotes the neural network's output with a logistic sigmoid as its activation function, i.e. $\mathrm{NN}=sigma(a) = 1/(1+exp(-a))$.

Weights and biases form the latent variables $\mathbf{w}$.

Define the prior on the weights and biases $\mathbf{w}$ to be the standard normal $$\begin{aligned} p(\mathbf{w}) &amp;= \text{Normal}(\mathbf{w} \mid \mathbf{0}, \mathbf{I}).\end{aligned}$$

Second, build this classification model in Edward. Again, we define a 3-layer Bayesian neural network with $\tanh$ nonlinearities

# References:

Neal, R. M. (2012). Bayesian learning for neural networks (Vol. 118). Springer Science & Business Media.



