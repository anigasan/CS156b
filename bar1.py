import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
#import cv2

df = pd.read_csv('/Users/anigasan/Desktop/cs156bstuff/CS156b/train/train.csv')

#Imputer code
#print(df.info())

#Image Paths


paths = list(df['Path'])
#above will only be used in the event of running on HPC
list_trains = os.listdir('/Users/anigasan/Desktop/cs156bstuff/CS156b/train')
training_paths = []


#Loaded locally
for train_dir in list_trains:
    if train_dir != 'train.csv':
        train_dir_l1 = os.listdir('/Users/anigasan/Desktop/cs156bstuff/CS156b/train' + '/' + train_dir)
        for train_dir1 in train_dir_l1:
            train_dir_l2 = os.listdir('/Users/anigasan/Desktop/cs156bstuff/CS156b/train' + '/' + train_dir + '/' + train_dir1)
            for train_dir2 in train_dir_l2:
                training_paths.append('train' + '/' + train_dir + '/' + train_dir1 + '/' + train_dir2)


#print(list_trains)

img_tensors = []
for i in range(len(training_paths)):
    training_paths[i] = '/Users/anigasan/Desktop/cs156bstuff/CS156b/' + training_paths[i]
    if '.jpg' in training_paths[i]:
        img = Image.open(training_paths[i])
        converter = transforms.ToTensor()
        tensor_img = converter(img)
        img_tensors.append(tensor_img)

    
    


    

    
    