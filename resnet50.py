import torch
import torchvision
from torchvision.models import resnet50
import torch.nn as nn
import os
import sys

def count_folders(directory):
    # Get list of all items (files and folders) in the directory
    items = os.listdir(directory)
    return len(items)
    
    # Filter out only the directories
    folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]
    
    # Return the number of folders
    return len(folders)

def stop_check():
    fg = open("run.txt")
    f = fg.read()
    fg.close()
    if f[0] == "0":
        return True
    else:
        return False

#hpc_path = "/groups/CS156b/2024/Edgemax_2/"
hpc_path = "/Users/tarunrajramgulam/Desktop/"
X_train_path = hpc_path + "X_train/"
y_train_path = hpc_path + "y_train/"
num_files = count_folders(X_train_path)
num_files = min(num_files, 100)
#print(num_files)
#sys.exit()


num_epochs = 1000

model = resnet50(pretrained = True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 256),
                         nn.ReLU(), 
                         nn.Linear(256, 9), #We'll add the necessary layers in here, 256 is used as a placeholder
                         nn.Tanh())

model.load_state_dict(torch.load('/Users/tarunrajramgulam/Desktop/resnet50_model.pth'))
model.to("mps")
    
criterion = torch.nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#Training code
model.train()
i = 1
f = open("run.txt", "w") # if you change the value of this file to "0", the training will stop
f.write("{0}".format(1))
f.close()
for epoch in range(num_epochs):
    print("Epoch = {}".format(epoch+1))
    j = 0
    total_loss = 0
    counter = 0
    for j in range(0, num_files): #Will fill this in later
        X = torch.load(X_train_path + "X_{}.pt".format(j+1))
        y = torch.load(y_train_path + "y_{}.pt".format(j+1))
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss
        counter += 1
        if stop_check():
            break
    print(total_loss/counter)
    if stop_check():
            break
    i += 1

torch.save(model.state_dict(), hpc_path + 'resnet50_model.pth')
