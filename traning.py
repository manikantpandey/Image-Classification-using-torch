# Importing  the required libraries

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms, models
import os
# Define data transformations 

data_transform= {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor( ),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])    
}

# Define Data directory
data_dir= 'dataset'

# Data Loders
image_datasets = {x: datasets.Imagefolder(os.path.joins(data_dir,x),data_transform[x])for x in ['train','val']}
dataloaders= {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)for x in ['train','val']}
class_names= image_datasets['train'].classes


# Load the pre-trained ResNet-18 model
model=models.resnet18(pretrained=True)

for name, param in model.named_parameters():
    if "fc" in name:
        param.requires_grad=True
    else:
        param.requires_grad=False

criterion= nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=model.to(device)





