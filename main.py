import numpy as np
import matplotlib.pyplot as plt
from network import *

# Loads images from the fashion mnist dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Defines the possible outputs
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Transforms the images to a greyscale
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshapes the sets to explicitly mention they have one 'color dimension' (grey scale)
train_images = train_images.reshape(len(train_images), train_images.shape[1], train_images.shape[2], 1)
test_images = test_images.reshape(len(test_images), test_images.shape[1], test_images.shape[2], 1)

