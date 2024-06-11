import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class NN (nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear (input_size, 50)
        self.fc2 = nn.Linear (50, num_classes)

    def forward (self, X):
        X = F.relu (self.fc1 (X))
        X = self.fc2 (X)
        return X
    
class CNN (nn.Module):
    def __init__ (self, in_channels = 1, out_channels = 10):
        super (CNN, self).__init__()
        self.conv1 = nn.Conv2d (in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d (kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d (in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear (16*7*7, num_classes)

    def forward (self, x):
        x = F.relu (self.conv1 (x))
        x = self.pool (x)
        x = F.relu (self.conv2 (x))
        x = self.pool (x)
        x = x.reshape (x.shape[0], -1)
        x = self.fc1 (x)
        return x
    
device = torch.device ('cuda' if torch.cuda.is_available () else 'cpu')

input_size, num_classes, alpha, batch_size, epochs = 1, 10, 0.001, 64, 1

train_dataset = datasets.MNIST (root='./dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST (root='./dataset/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader (dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader (dataset=test_dataset, shuffle=True, batch_size=batch_size)

model = CNN (in_channels=input_size, out_channels=num_classes).to(device)

criterion = nn.CrossEntropyLoss ()
optimizer = torch.optim.Adam (model.parameters (), lr=alpha)

for epoch in range (epochs):
    for batch_idx, (data, targets) in enumerate (train_loader):
        data = data
        target = targets

        y_pred = model (data)
        loss = criterion (y_pred, targets)

        loss.backward ()
        optimizer.step ()

        optimizer.zero_grad ()

def check_accuracy (loader, model):
    num_correct, num_samples = 0, 0
    model.eval ()

    with torch.no_grad ():
        for x, y in loader:
            scores = model (x)
            _, predictions = scores.max (1)
            num_correct += (predictions == y).sum ()
            num_samples += predictions.size (0)

        print (f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train ()


check_accuracy (train_loader, model)
check_accuracy (test_loader, model)

