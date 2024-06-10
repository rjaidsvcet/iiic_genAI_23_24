import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

X_data, y_data = datasets.make_regression (n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy (X_data.astype (np.float32))
y = torch.from_numpy (y_data.astype (np.float32))
y = y.view (y.shape[0], 1)

print (len (X))

n_samples, n_features = X.shape

input_size = n_features
output_size = 1

# class LinearRegression (nn.Module):
#     def __init__(self, input_dimension, output_dimension):
#         super (LinearRegression, self).__init__()
#         self.linear = nn.Linear (input_dimension, output_dimension)

#     def forward (self, X):
#         return self.linear (X)

# model = LinearRegression (input_dimension = input_size, output_dimension = output_size)

model = nn.Linear (input_size, output_size)

alpha = 0.01
epochs = 100

criterion = nn.MSELoss ()
optimizer = torch.optim.SGD (model.parameters (), lr = alpha)


for epoch in range (epochs):
    y_pred = model (X)
    loss = criterion (y_pred, y)
    loss.backward ()
    optimizer.step ()
    optimizer.zero_grad ()
    if (epoch + 1) % 10 == 0:
        print (f'Epoch : {epoch + 1}, Loss : {loss.item ():.4f}')

predicted = model (X).detach ().numpy ()
plt.plot (X_data, y_data, 'ro')
plt.plot (X_data, predicted, 'b')
plt.show ()