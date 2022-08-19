from pickletools import optimize
from matplotlib import testing
import torch 
from torch import nn, optim
from torch.utils.data import dataloader
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

print(" ")
training_transforms = transforms.Compose([
    transforms.RandomResizedCrop(28),                    # randomly zoomout, zoomin and crop
    transforms.RandomRotation(45),
    transforms.ToTensor() 
    ])

training_data = datasets.MNIST(
    root = "data",
    train=True,
    download=True,
    transform=training_transforms 
)

print(len(training_data))


feature = training_data[432][0]
label = training_data[432][1]

print(label)
print(feature.size)
print(feature)

plt.imshow(feature)
plt.show()

print(" ")