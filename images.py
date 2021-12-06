import numpy as np
import skimage.io as io
from pylab import *
import cv2
from skimage import feature, morphology, filters, util
from matplotlib import pyplot as plt
from matplotlib import gridspec as grd
import matplotlib.pyplot as plt
from PIL import Image

def get_file_path(group_name, number, processed=0):
    if processed == 1:
        file_path = "img_processed/" + group_name + '/' + group_name
        number = number - 1
    elif processed == 2:
        file_path = "img_test/" + group_name + '/test_' + group_name
    elif processed == 3:
        file_path = "img_test_processed/" + group_name + '/' + group_name
        number = number - 1
    else:
        file_path = "img_to_teach/" + group_name + '/' + group_name
    if number < 10:
        file_path += '0'
    file_path += str(number) + ".jpg"
    return file_path


def load_photos_from_group(group_name, processed=0):
    group_photos = []
    if processed > 1:
        n = 9
    else:
        n = 25
    for i in range(n):
        file_path = get_file_path(group_name, i+1, processed)
        image = io.imread(file_path, as_gray=True)
        group_photos.append(image)
    return group_photos


def load_groups(group_names, processed=0):
    groups = []
    for g in group_names:
        single_group = load_photos_from_group(g, processed)
        groups.append(single_group)
    return groups

def preprocess_group(group, g_name, processed=0):

    for i, v in enumerate(group):
        image = v.copy()
        image = feature.canny(image=image, sigma=1.5)
        if processed == 0:
            file_path = "img_processed/" + g_name + '/' + g_name
        else:
            file_path = "img_test_processed/" + g_name + '/' + g_name
        if i < 10:
            file_path += '0'
        file_path += str(i) + ".jpg"
        io.imsave(file_path, util.img_as_ubyte(image))

def preprocess(groups, names, processed=0):
    for i, group in enumerate(groups):
        preprocess_group(group, names[i], processed)

def calculate_hu(image):
    moments = cv2.moments(image)
    huMoments = cv2.HuMoments(moments)
    return huMoments

def normalize_hu(moments):
    for i in range(7):
        moments[i] = -1* copysign(1.0, moments[i])*log10(abs(moments[i]))
    return moments

def make_desc(photos, names, processed=0):
    desc = list()
    if processed == 0:
        n = 25
    else:
        n = 9
    for i, group in enumerate(names):
        for j in range(n):
            desc.append((normalize_hu(calculate_hu(photos[i][j])), group))
    return desc

def dist_desc(desc1, desc2):
    dist = 0
    for i in range(len(desc1)):
        dist += (desc1[i]-desc2[i])**2
    return sqrt(dist)

def prediction(data, test, k):
    dists = list()
    for dset in data:
        dists.append((dset[1], dist_desc(dset[0], test)))
    dists.sort(key=lambda t: t[1])
    nghb = dict()
    for i in range(k):
        name = dists[i][0]
        if name in nghb.keys():
            nghb[name] += 1
        else:
            nghb[name] = 1
    v = max(nghb.values())
    name = list()
    for i in nghb.keys():
        if nghb[i] == v:
            name.append(i)
    return name

def select_k(data):
    max_match = 0
    k = 1
    for k_guess in range(3, 21):
        matches = 0
        for i in data:
            predict = prediction(data, i[0], k_guess)
            if predict[0] == i[1]:
                matches += 1
        if matches > max_match:
            max_match = matches
            k = k_guess
    return k

def teaching_process(names):
    photos = load_groups(names)

    preprocess(photos, names)

    photos = load_groups(names, processed=1)

    data = make_desc(photos, names)
    return (data, select_k(data))

def prepare_test(names):
    data = load_groups(names, 2)
    preprocess(data, names, 1)

    data = load_groups(names, 3)
    moment = make_desc(data, names, 1)

    return moment

def make_test(k, test, data, names):
    res = dict()
    for i in names:
        res[i] = dict()
        for j in names:
            res[i][j] = 0
    for i in test:
        predict = prediction(data, i[0], k)
        if i[1] in predict:
            res[i[1]][i[1]] += 1
        else:
            res[i[1]][predict[0]] += 1
    for i in data:
        predict = prediction(data, i[0], k)
        if i[1] in predict:
            res[i[1]][i[1]] += 1
        else:
            res[i[1]][predict[0]] += 1
    return res


if __name__ == '__main__':
    names = ("apple", "asus", "dell", "hp", "huawei", "microsoft")
    data_set, k = teaching_process(names)
    test_set = prepare_test(names)
    matrix = make_test(k, test_set, data_set, names)
    print(end="           ")

    for i in matrix.keys():
        print(i, end="    ")
    print()
    for i in matrix.keys():
        print(i, end="         ")
        for j in matrix[i].keys():
            print(str(matrix[i][j]/34 * 100) + '%', end="     ")
        print()
