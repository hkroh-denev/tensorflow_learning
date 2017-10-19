import os
import numpy as np
from image_util import save_image, save_images

def load_mnist(path):
    fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
    images = np.fromfile(file=fd, dtype=np.uint8)
    images = images[16:].reshape([60000, 28, 28]).astype(np.float)

    fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
    labels = np.fromfile(file=fd, dtype=np.uint8)
    labels = labels[8:].reshape([60000]).astype(np.float)

    return images, labels

# TEST DRIVE
images, labels = load_mnist('./mnist')
save_images('output.png', images[0:64], [8, 8])
