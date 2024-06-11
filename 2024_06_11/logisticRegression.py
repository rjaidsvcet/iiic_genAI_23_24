from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import numpy as np


data = load_breast_cancer ()
X, y = data.data, data.target
# print (len (X))

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=1234)

sc = StandardScaler ()
X_train = sc.fit_transform (X_train)
X_test = sc.transform (X_test)

X_train = torch.from_numpy (X_train.astype (np.float32))
X_test = torch.from_numpy (X_test.astype (np.float32))
y_train = torch.from_numpy (y_train.astype (np.float32))
y_test = torch.from_numpy (y_test.astype (np.float32))

y_train = y_train.view (y_train.shape[0], 1)
y_test = y_test.view (y_test.shape[0], 1)

class LogisticRegession (nn.Module):
    def __init__ (self, input_features):
        super (LogisticRegession, self).__init__()
        self.first_layer = nn.Linear (input_features, 1)

    def forward (self, X):
        y_pred = torch.sigmoid (self.first_layer (X))
        return y_pred
    
n_samples, n_features = X.shape
model = LogisticRegession (input_features=n_features)

alpha = 0.01
epochs = 100
criterion = nn.BCELoss ()
optimizer = torch.optim.SGD (model.parameters (), lr=alpha)

for epoch in range (epochs):
    y_pred = model (X_train)
    loss = criterion (y_pred, y_train)
    loss.backward ()
    optimizer.step ()
    optimizer.zero_grad ()
    if (epoch + 1) % 10 == 0:
        print (f'Epoch : {epoch+1} and Loss : {loss.item():.5f}')

with torch.no_grad ():
    y_hat = model (X_test)
    y_hat_classified = y_hat.round ()
    accuracy = y_hat_classified.eq  (y_test).sum() / float (y_test.shape[0])
    print (f'Accuracy : {accuracy:.4f}')