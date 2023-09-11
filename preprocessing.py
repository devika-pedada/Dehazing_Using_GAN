from os import listdir
from numpy import asarray, load
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
from matplotlib import pyplot
import numpy as np


# def load_images(path1, path2, size=(256, 256)):
#     gt_list, hazy_list = list(), list()
#     # enumerate filenames in directory, assume all are images
#     for filename in listdir(path1):
#         # load and resize the image
#         pixels = load_img(path1 + filename, target_size=size)
#         # convert to numpy array
#         gt_img = img_to_array(pixels)
#         gt_list.append(gt_img)
#
#     for filename in listdir(path2):
#         # load and resize the image
#         pixels = load_img(path2 + filename, target_size=size)
#         # convert to numpy array
#         hazy_img = img_to_array(pixels)
#         hazy_list.append(hazy_img)
#     return [asarray(hazy_list), asarray(gt_list)]


def load_images(path_hazy, path_gt, size=(256, 256)):
    gt_list, hazy_list = list(), list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path_hazy):
        # load and resize the image
        pixels = load_img(path_hazy + filename, target_size=size)
        # convert to numpy array
        hazy_img = img_to_array(pixels)
        hazy_list.append(hazy_img)

    for filename in listdir(path_gt):
        # load and resize the image
        pixels = load_img(path_gt + filename, target_size=size)
        # convert to numpy array
        gt_img = img_to_array(pixels)
        gt_list.append(gt_img)
    return [asarray(hazy_list), asarray(gt_list)]


def load_single_image(hazy_img, gt_img, size=(256, 256)):
    hazy_list, gt_list = list(), list()

    pixels = load_img(hazy_img, target_size=size)
    hazy_array = img_to_array(pixels)
    hazy_list.append(hazy_array)

    pixels = load_img(gt_img, target_size=size)
    gt_array = img_to_array(pixels)
    gt_list.append(gt_array)

    return [asarray(hazy_list), asarray(gt_list)]




def preprocess_data(data):
    # load compressed arrays
    # unpack arrays
    X1, X2 = data[0], data[1]
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]