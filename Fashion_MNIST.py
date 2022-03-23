import torch 
from torch import nn, optim
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                          ])
trainset = datasets.FashionMNIST('~/.pytorch/Fashion-MNIST/', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('~/.pytorch/Fashion-MNIST/', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim = 1)
        
        return x
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.005)
epochs = 1
steps = 0
train_losses, test_losses = [], []
for i in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        logps = model(images)
        loss = criterion(logps, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                logps = model(images)
                test_loss+=criterion(logps, labels)
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
        model.train()
        print('Epoch: {}/{}'.format(i+1, epochs), 
              'Training loss: {:.3f}.. '.format(running_loss/len(trainloader)), 
              'Test loss : {:.3f}.. '.format(test_loss/len(testloader)),
              'Test accuracy: {:.3f}.. '.format(accuracy/len(testloader)))
plt.plot(train_losses, label = 'Training loss')
plt.plot(test_losses, label = 'Test loss')
plt.legend(frameon = False)
plt.show()
torch.save(model.state_dict(), 'checkpoint.pth')
state_dict = torch.load('checkpoint.pth')
print(state_dict)
#loading state_dict to model
model.load_state_dict(state_dict)
checkpoint = {'input_size' : 784,
              'output_size' : 10, 
              'hidden_layers' : [each.out_features for each in model.hidden_layers],
              'state_dict' : model.state_dict}
torch.save(checkpoint, 'checkpoint.pth')
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = Classifier(checkpoint['input_size'],
                               checkpoint['output_size'],
                               checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    return model
model = load_checkpoint('checkpoint.pth')
print(model)