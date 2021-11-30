import numpy as np
import skimage.io as io
from pylab import *
import cv2
from skimage import feature, morphology, filters, util
from matplotlib import pyplot as plt
from matplotlib import gridspec as grd
import matplotlib.pyplot as plt
from PIL import Image

def get_file_path(group_name, number):
    file_path = "img_processed/" + group_name + '/' + group_name
    number = number-1 ### for processed
    if number < 10:
        file_path += '0'
    file_path += str(number) + ".jpg"
    return file_path


def load_photos_from_group(group_name):
    group_photos = []
    for i in range(25):
        file_path = get_file_path(group_name, i+1)
        image = io.imread(file_path, as_gray=True)
        # print("Loaded "+group_name+" no. \t"+str(i+1))
        group_photos.append(image)
    return group_photos


def load_groups(group_names):   # group_names = ("apple", "asus", "dell", "hp", "huawei", "microsoft")
    groups = []
    for g in group_names:
        single_group = load_photos_from_group(g)
        groups.append(single_group)
    return groups

def preprocess_group(group, g_name):

    for i, v in enumerate(group):
        image = v
        #image = morphology.erosion(image, morphology.square(4))
        #image = morphology.dilation(image, morphology.square(3))
        #image = filters.rank.median(util.img_as_ubyte(image), ones([3, 3], dtype=uint8))
        image = feature.canny(image=image, sigma=1.5)

        file_path = "img_processed/" + g_name + '/' + g_name
        if i < 10:
            file_path += '0'
        file_path += str(i) + ".jpg"
        io.imsave(file_path, util.img_as_ubyte(image))

def preprocess(groups, names):
    for i, group in enumerate(groups):
        preprocess_group(group, names[i])

def calculate_hu(image):
    moments = cv2.moments(image)
    huMoments = cv2.HuMoments(moments)
    return huMoments

def normalize_hu(moments):
    for i in range(7):
        moments[i] = -1* copysign(1.0, moments[i])*log10(abs(moments[i]))
    return moments

def make_desc(photos, names):
    desc = list()
    for i, group in enumerate(names):
        for j in range(25):
            desc.append((normalize_hu(calculate_hu(photos[i][j])), group))
    return desc

names = ("apple", "asus", "dell", "hp", "huawei", "microsoft")
photos = load_groups(names)

#preprocess(photos, names)
data = make_desc(photos, names)
