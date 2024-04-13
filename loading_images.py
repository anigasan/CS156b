import numpy as np
import torch
from PIL import Image
import pandas as pd
import os
from torchvision import transforms
import torchvision


label_file = 'train.csv'
df = pd.read_csv(label_file)

#print(df["Path"][0])
#print(df.shape[0])

def load_image(path):
    b = torchvision.io.read_image(path).to(torch.float32)
    return b

#print(load_image("pid01310/study1/view1_frontal.jpg"))

direction = "Frontal"
AP_PA = "AP"
store_path = "{}/{}/Model_1".format(direction, AP_PA)
#labels = [["No Finding"]]
labels = list(df.columns)
print(labels)
#def gen_tensors()

