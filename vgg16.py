import torch
import torchvision
from torchvision.models import resnet50


num_epochs = 50

model = resnet50(pretrained = True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 256),
                         nn.ReLU(), 
                         nn.Linear(256, 9)) #We'll add the necessary layers in here, 256 is used as a placeholder
    
criterion = torch.nn.MSELoss()
optimizer = torch.nn.Adam(model.parameters(), lr=0.001)


#Training code
model.train()
for epoch in range(num_epochs):
    for input_val, label in placeholder: #Will fill this in later
        output = model(input_val)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    