import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class NN (nn.Module):
    def __init__ (self, input_size, num_classes):
        super (NN, self).__init__()
        self.fc1 = nn.Linear (input_size, 50)
        self.fc2 = nn.Linear (50, num_classes)

    def forward (self, X):
        X = F.relu (self.fc1 (X))
        X = self.fc2 (X)
        return X
    
input_size, num_classes = 784, 10
alpha, epochs, batch_size = 0.001, 1, 64

train_dataset = datasets.MNIST (root='./dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST (root='./dataset/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader (dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader (dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = NN (input_size=input_size, num_classes=num_classes)

criterion = nn.CrossEntropyLoss ()
optimizer = torch.optim.Adam (model.parameters (), lr=alpha)

for epoch in range (epochs):
    for batch_idx, (data, targets) in enumerate (train_loader):
        data = data
        target = targets
        data = data.reshape (data.shape[0], -1)
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
            x = x.reshape (x.shape[0], -1)
            scores = model (x)
            _, predictions = scores.max (1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size (0)
        print (f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

        model.train ()

check_accuracy (train_loader, model)
check_accuracy (test_loader, model)
