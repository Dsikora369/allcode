import torch 
from torch import nn, optim
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])
train_data = datasets.ImageFolder('train/', transform = train_transforms)
test_data = datasets.ImageFolder('validation/', transform = test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 32)
model = models.densenet121(pretrained = True)
#print(model)
for param in model.parameters():
    param.requires_grad = False
    
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
    ('fcl', nn.Linear(1024, 500)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500, 2)),
    ('output', nn.LogSoftmax(dim=1))
    ]))
model.classifier = classifier
for cuda in [False]:
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
    if cuda:
        model.cuda()
    else:
        model.cpu()
    for ii, (inputs, labels) in enumerate(trainloader):
        inputs, labels = Variable(inputs), Variable(labels)
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        start = time.time()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if ii == 3:
            break
    print(f'CUDA = {cuda}; Time per batch: {time.time() - start/3:.3f} seconds')