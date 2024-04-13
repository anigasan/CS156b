import torch
import torchvision
from torchvision.models import resnet50



lung_model = resnet50(pretrained = True)
criterion = torch.nn.MSELoss()
optimizer = torch.nn.Adam(lung_model.parameters(), lr=0.001)


#for epoch in range(100):
    
    