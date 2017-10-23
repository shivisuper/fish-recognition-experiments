# fish-recognition-experiments

This repository contains the various experiments I performed using Deep Learning to implement a fish recognizer.  
To start with your own experiments, make sure to follow steps below to install all the prerequisites

1. Platform - Ubuntu 16.04
2. Python version - 3.5

### Libraries

I would recommend installing __VirtualEnv__ and __VirtualEnvWrapper__ to maintain different python environments .
Before running the scripts, your python environment should have following libraries installed .
Having different python environments is a good choice if you're working on a large project . 
Also, you don't need "sudo" to install following libraries if you're installing them in virtual environment

* Numpy (use pip to install latest version)
* Scikit-Learn (use pip to install latest version)
* TensorFlow (build from source, refer [this](https://www.tensorflow.org/install/install_sources "TensorFlow build from source"))
* OpenCV 3.3 ([this](http://milq.github.io/install-opencv-ubuntu-debian/) is an excellent guide to install opencv)
* MatPlotlib (use pip to install latest version)
* Keras (use pip to install latest version)

_Note_: When using Keras please refer their [documentation](https://keras.io/#installation "Keras docs") where 
they explain how to configure the backend engine (TensorFlow or Theano). I have used TensorFlow for all experiments

### Running scripts

Until specified otherwise inside the script itself, running scripts is pretty straight forward by calling the python  
program with the script name. Again, if you're using __VirtualEnv__ then remember to execute `workon <envname>`  
before executing the scripts.
