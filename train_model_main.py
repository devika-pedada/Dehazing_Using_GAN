from os import listdir
from numpy import asarray, load
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
from matplotlib import pyplot
import numpy as np
from preprocessing import load_images, preprocess_data
from pix2pix_model import define_discriminator, define_generator, define_gan, train
from datetime import datetime


# dataset path
path_gt = 'O-HAZY/GT/'
path_hazy = 'O-HAZY/hazy/'
# load dataset
# [src_images, tar_images] = load_images(path)
[src_images, tar_images] = load_images(path_hazy, path_gt)
print('Loaded: ', src_images.shape, tar_images.shape)

n_samples = 3
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(src_images[i].astype('uint8'))
# plot target image
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + n_samples + i)
    pyplot.axis('off')
    pyplot.imshow(tar_images[i].astype('uint8'))
pyplot.show()

#######################################

# define input shape based on the loaded dataset
image_shape = src_images.shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

# Define data
# load and prepare training images
data = [src_images, tar_images]
dataset = preprocess_data(data)

#########################################

start1 = datetime.now()

train(d_model, g_model, gan_model, dataset, n_epochs=10, n_batch=1)
# Reports parameters for each batch (total 1096) for each epoch.
# For 10 epochs we should see 10960

stop1 = datetime.now()
# Execution time of the model
execution_time = stop1 - start1
print("Execution time is: ", execution_time)

# Reports parameters for each batch (total 1096) for each epoch.
# For 10 epochs we should see 10960

#########################################


