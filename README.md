# Technical Report: Exploiting Deep-Features Diversity in Food-11 Classification

This is the source-code of technical report using the Food-11 dataset. In next steps, all necessary information to perform the experiments will be presented.


## Prerequisites

The experiments were built using **Python 2.7.12**, with some libraries: **scikit-learn, theano, keras, numpy, scipy,  matplotlib, Pillow and h5py**. It is required to install these libraries to run the code.

* [scikit-learn](http://scikit-learn.org/stable/) - Machine Learning library for Python
* [Theano](http://deeplearning.net/software/theano/) - Deep Learning toolkit
* [Keras](https://keras.io/) - High-level API for Deep learning toolkits
* [numpy](http://www.numpy.org/) - Scientific computing library for Python
* [scipy](https://www.scipy.org/) -  Ecosystem of open-source software for mathematics, science, and engineering
* [matplotlib](http://matplotlib.org/) - 2D plotting library
* [Pillow](https://github.com/python-pillow/Pillow) - Python Image Library Fork
* [h5py](http://www.h5py.org/) - HDF5 for Python

All of these libraries could be installed using [pip](https://pypi.python.org/pypi/pip) (Python Package Index).

```
sudo -H pip install scikit theano keras numpy scipy matplotlib Pillow h5py
```

[Python-Tk](https://docs.python.org/2/library/tkinter.html) must be installed too, to render the GUI with python. To install it, just execute
```
sudo apt-get install python-tk
```

After installing them, we can run the experiments.

## Dataset

The dataset used in our experiments is the [Food-11](http://grebvm2.epfl.ch/lin/food/Food-11.zip) dataset. Food-11 contains 16643 food images grouped in 11 major food categories. The 11 categories are Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit. The dataset is divided on ***training, validation and testing***.

The total file size of the Food-11 dataset is about 1.16 GB.

> Download it and put in the same directory of source-code.

### AlexNet

We modified a TensorFlow version of AlexNet, found on [Heuritech Github](https://github.com/heuritech/convnets-keras). To the feature extraction work, the ImageNet weights must be [downloaded](http://files.heuritech.com/weights/alexnet_weights.h5).

> Put it on same directory of source-code.

## Running

The experiments consists of three sequentially, but separated process:

1. Mount subsampled dataset
2. Extract features using pre-trained AlexNet
3. Choose and execute one of available experiments

The first two steps should be executed once. The third step comprises a set of experiments that can be executed as long as needed.

### Mounting subsampled dataset

Just run the `build_dataset.py` script. In the script, **folder** variable chooses where to get images and **max_size** defines the amount of images used.

An array in *numpy* format (trab2_dataset.npz) will be saved on **data** folder.

### Extracting features using pre-trained AlexNet

Just run the `feature_extraction.py` script. Three numpy arrays will be created on **data** folder (trab2_conv1.npz, trab2_conv5.npz and trab2_dense2.npz).

### Running the experiments

Three experiments are available:

* Deep Features: Classification over C1, C2 and FC2 layers from AlexNet (`feat_experiment.py`)
* Early and Late Fusion: Classification using Early and late fusion approaches (`early_fusion_experiment.py` and `late_fusion_experiment.py`)
* Ensemble: Diversity tests using Random Forest, Majority Vote and Bagging (`ensemble_experiment.py`)


## Authors

* **Alan C. Neves**
* **Caio C. V. da Silva**
* **Jéssica Soares**
* **Sérgio J. de Sousa**

