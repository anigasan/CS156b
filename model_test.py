import torch
import torchvision
from torchvision.models import resnet50
import torch.nn as nn
import os
import sys
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd

hpc_path = "/Users/tarunrajramgulam/Desktop/"
transform_image = transforms.Compose([transforms.Resize((224, 224)), 
                            transforms.ToTensor()])

model = torchvision.models.resnet50(pretrained=True)  # Initialize the model
model.fc = nn.Sequential(nn.Linear(2048, 256),
                         nn.ReLU(), 
                         nn.Linear(256, 9), #We'll add the necessary layers in here, 256 is used as a placeholder
                         nn.Tanh())
model.load_state_dict(torch.load('/Users/tarunrajramgulam/Desktop/resnet50_model.pth'))
model.to("mps")
model.eval()  # Set the model to evaluation mode

def process_image(im): # returns a PyTorch Tensor
    return im

def load_image(path, device="cpu"):
    #b = torchvision.io.read_image(path).to(torch.float32)
    im = Image.open(path)
    im = process_image(im)
    b = transform_image(im)
    b = b.to(device)
    b = b.unsqueeze(0)
    b = b.expand(1, 3, -1, -1)
    return b[0]

def get_folders(directory, keyword=None):
    folders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            if keyword is None:
                folders.append(item)
            elif keyword in item:
                folders.append(item)
    return folders

def get_ids(pid_folder_list):
    id_list = []
    for f1 in pid_folder_list:
        id_list.append(f1[3:])
    return id_list

def get_image_paths(path):
    f1 = get_folders(path, keyword="pid")
    ids = get_ids(f1)
    num_ids = len(ids)
    output_array = np.zeros((num_ids, 9))
    #output_array[:, 0] = ids

    counter = 0
    for folder in f1:
        new_path = path + folder + "/"
        f2 = get_folders(new_path, keyword="study")
        k = len(f2)
        input_tensor = torch.zeros((k, 3, 224, 224))
        counter_2 = 0
        for i in range(0, k):
            study = new_path + f2[i] + "/"
            im_path = study + "view1_frontal.jpg"
            if os.path.exists(im_path):
                input_tensor[counter_2] = load_image(study + "view1_frontal.jpg")
                counter_2 += 1
        if counter_2 > 0:
            input_tensor = input_tensor[:counter_2]
        output = model(input_tensor).detach().clone()
        output_array[counter, :] = torch.mean(output, axis=0)
        counter += 1
        print(counter/num_ids)
    return output_array, ids

def test_2(test_path, test_ids, batch_size=32, device="cpu"):
    if len(test_path) != 0 and test_path[-1] != '/':
        test_path = test_path + '/'
    ids = list(test_ids["Id"])
    #ids = None
    paths = list(test_ids["Path"])
    n = len(paths)
    #n = 100
    ids = ids[:n]
    counter = 0
    output_array = torch.zeros((n, 9), device=device)
    for i in range(0, n):
        im_path = paths[i]
        if counter == 0:
            current_ix = i
            K = min(batch_size, n-i)
            im_batch = torch.zeros((K, 3, 224, 224),device=device)
        im_batch[counter] = load_image(test_path + im_path, device=device)
        if counter == K - 1:
            #im_batch = im_batch.to("mps")
            output = model(im_batch)
            #print(torch.linalg.norm(output))
            output_array[current_ix:current_ix + K, :] = output.detach().clone()
            counter = 0
        else:
            counter += 1
        print(i/n)
    return output_array.to("cpu"), ids

#arrs, ids = get_image_paths(hpc_path)
test_ids = pd.read_csv(hpc_path + "train/test_ids.csv")
arrs, ids = test_2(hpc_path, test_ids, batch_size=32, device="mps")
#print(arrs)
df = pd.DataFrame(arrs, columns=['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Pneumonia','Pleural Effusion','Pleural Other','Fracture','Support Devices'])
df.insert(0, 'Id', ids)
df.to_csv('submission.csv', index=False)
