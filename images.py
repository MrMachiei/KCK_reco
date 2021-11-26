import numpy as np
from skimage import data, io, feature, transform, color
import matplotlib.pyplot as plt


def get_file_path(group_name, number):
    file_path = "img_to_teach/" + group_name + '/' + group_name
    if number < 10:
        file_path += '0'
    file_path += str(number) + ".jpg"
    return file_path


def load_photos_from_group(group_name):
    group_photos = []
    for i in range(25):
        file_path = get_file_path(group_name, i+1)
        image = io.imread(file_path)
        # print("Loaded "+group_name+" no. \t"+str(i+1))
        group_photos.append(image)
    return group_photos


def load_groups(group_names):   # group_names = ("apple", "asus", "dell", "hp", "huawei", "microsoft")
    groups = []
    for g in group_names:
        single_group = load_photos_from_group(g)
        groups.append(single_group)
    return groups


names = ("apple", "asus", "dell", "hp", "huawei", "microsoft")
photos = load_groups(names)
for g in photos:
    # counter = 0
    for img in g:
        plt.imshow(img)
        plt.show()
        # counter += 1
        # print(counter)
