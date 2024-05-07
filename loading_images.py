import numpy as np
import torch
from PIL import Image
import pandas as pd
import os
from torchvision import transforms
import torchvision
from sklearn.impute import SimpleImputer
import sys


#label_file = '/groups/CS156b/data/student_labels/train2023.csv'
label_file = 'train2023.csv'
df = pd.read_csv(label_file)
#print(df)
#sys.exit()
#df = df.fillna(0)
classes = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Pneumonia','Pleural Effusion','Pleural Other','Fracture','Support Devices']

imputer_floats = SimpleImputer(strategy='mean')
df_floats = df.select_dtypes('float64')
df_floats_final = pd.DataFrame(df_floats[classes])
df_floats_imputed = pd.DataFrame(imputer_floats.fit_transform(df_floats_final), columns = df_floats_final.columns)
df_non_floats = df.select_dtypes(include = ['int64', 'object'])
df = pd.concat([df_non_floats, df_floats_imputed], axis=1)

#print(df)
#print(df["Path"][0])
#print(df.shape[0])

transform_image = transforms.Compose([transforms.Resize((224, 224)), 
                            transforms.ToTensor()])

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

def gen_training_data(data_path, final_path, df, classes, image_dim = (224, 224), batch_size=32, device="cpu"):
    batch_tensor_size = ()
    all_paths = list(df['Path'])
    n = len(all_paths)
    if len(data_path) != 0 and data_path[-1] != '/':
        data_path = data_path + '/'

    if len(final_path) != 0 and final_path[-1] != '/':
        final_path = final_path + '/'
    
    ix = 0
    TensorCount = 1
    num_classes = len(classes)
    destination_X = final_path + "X_train/"
    destination_y = final_path + "y_train/"
    if not os.path.exists(destination_X):
        os.makedirs(destination_X)
    if not os.path.exists(destination_y):
        os.makedirs(destination_y)

    y_tensor = torch.from_numpy(np.float32(np.array(df[classes])))

    for i in range(0, n):
        if ix == 0:
            z_dim = min(batch_size, n-i)
            batch_tensor_size = (z_dim, 3, image_dim[0], image_dim[1])
            batch_tensor_X = torch.zeros(batch_tensor_size, device=device)
            batch_tensor_y = torch.zeros(z_dim, num_classes, device=device)
        image_path = all_paths[i]
        complete_path = data_path + image_path
        k = 0
        if os.path.exists(complete_path):
            try:
                im = load_image(complete_path, device=device)
                k = 1
            except:
                k = 0
        if k == 1:
            #batch_tensor_X[ix] = load_image(complete_path)
            batch_tensor_X[ix] = im
            batch_tensor_y[ix] = y_tensor[i]
            if ix + 1 == batch_size:
                torch.save(batch_tensor_X, destination_X + "X_{}.pt".format(TensorCount))
                torch.save(batch_tensor_y, destination_y + "y_{}.pt".format(TensorCount))
                TensorCount += 1
                ix = 0
            else:
                ix += 1
        print(i/n)

data_path = "/Users/tarunrajramgulam/Desktop/train/"
final_path = "/Users/tarunrajramgulam/Desktop/"

#data_path = "/groups/CS156b/data/"
#final_path = "/groups/CS156b/2024/Edgemax_2/"

gen_training_data(data_path, final_path, df, classes, batch_size=32, device="cpu")

hpc_path = "/groups/CS156b/2024/Edgemax_2"