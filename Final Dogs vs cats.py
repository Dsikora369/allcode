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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained = True)
for param in model.parameters():
    param.requires_grad = False
classifier = nn.Sequential(nn.Linear(2048, 512),
                           nn.ReLU(),
                           nn.Dropout(p = 0.2),
                           nn.Linear(512, 2),
                           nn.LogSoftmax(dim=1))
model.fc =  classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr = 0.003)
model.to(device)
epochs = 1
steps = 0
running_loss = 0
print_every = 1
for epoch in range(epochs):
    for images, labels in trainloader:
        steps+=1
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                logps = model(images)
                loss = criterion(logps, labels)
                test_loss+=loss.item()
                
                ps = torch.exp(logps)
                top_ps, top_class = ps.topk(1, dim = 1)
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor))
            print(f"Epochs {epoch + 1}/{epochs}.."
                  f"Train loss: {running_loss/print_every:.3f}.."
                  f"Test loss: {test_loss/len(testloader):.3f}.."
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
        