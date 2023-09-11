#Test trained model on a few images...

from keras.models import load_model
from numpy.random import randint
from numpy import vstack
from matplotlib import pyplot
# from p2p_satellite_maps import *
from preprocessing import load_images, preprocess_data, load_single_image


# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'Expected']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 3, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# show title
		pyplot.title(titles[i])
	pyplot.show()


#load images from a directory
tar_img_path = 'Testing_Images/GT/03_GT.png'
src_img_path = 'Testing_Images/hazy/03_hazy.png'

[src_images, tar_images] = load_single_image(src_img_path, tar_img_path)
data = [src_images, tar_images]
dataset = preprocess_data(data)

model = load_model('model_31_08_23_03.h5')

[X1, X2] = dataset

# select random example
# ix = randint(0, len(X1), 1)
# print(ix)
src_image, tar_image = X1, X2

# generate image from source
gen_image = model.predict(src_image)

# plot all three images
plot_images(src_image, gen_image, tar_image)

# for ix in range (1, len(X1)):
# 	src_image, tar_image = X1[ix], X2[ix]
# 	# generate image from source
# 	gen_image = model.predict(src_image)
# 	# plot all three images
# 	plot_images(src_image, gen_image, tar_image)





# #load images from a directory
# tar_img_path = 'Testing_Images/GT/'
# src_img_path = 'Testing_Images/hazy/'
#
# [src_images, tar_images] = load_images(src_img_path, tar_img_path)
# data = [src_images, tar_images]
# dataset = preprocess_data(data)
#
# model = load_model('model_31_08_23.h5')
#
# [X1, X2] = dataset
#
# # select random example
# ix = randint(0, len(X1), 1)
# print(ix)
# src_image, tar_image = X1[ix], X2[ix]
#
# # generate image from source
# gen_image = model.predict(src_image)
#
# # plot all three images
# plot_images(src_image, gen_image, tar_image)
#
# # for ix in range (1, len(X1)):
# # 	src_image, tar_image = X1[ix], X2[ix]
# # 	# generate image from source
# # 	gen_image = model.predict(src_image)
# # 	# plot all three images
# # 	plot_images(src_image, gen_image, tar_image)
#
#
