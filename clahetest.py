import cv2
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
from analysis_functions import *
import sys
from PIL import Image
import random
import skimage.exposure as skie
import sys
import os

list_trains = os.listdir('/Users/anigasan/Desktop/cs156bstuff/CS156b/train')
training_paths = []

for train_dir in list_trains:
    if train_dir != 'train.csv':
        train_dir_l1 = os.listdir('/Users/anigasan/Desktop/cs156bstuff/CS156b/train' + '/' + train_dir)
        for train_dir1 in train_dir_l1:
            train_dir_l2 = os.listdir('/Users/anigasan/Desktop/cs156bstuff/CS156b/train' + '/' + train_dir + '/' + train_dir1)
            for train_dir2 in train_dir_l2:
                training_paths.append('train' + '/' + train_dir + '/' + train_dir1 + '/' + train_dir2)


beginning = '/Users/anigasan/Desktop/cs156bstuff/CS156b/'
output_files = []

for i in range(len(training_paths)):
    if '.jpg' in training_paths[i]:
        img = image_parse(beginning + training_paths[i])
        clahe = cv2.createCLAHE(clipLimit = 80, tileGridSize = (10, 10))
        clahe_img = clahe.apply(img)
        im2 = sobel(clahe_img)
        output_files.append(im2)
