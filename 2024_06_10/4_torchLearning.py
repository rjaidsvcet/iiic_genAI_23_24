import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

n_samples, n_features = X.shape
print (f'Sample : {n_samples} and features : {n_features}')

X_test = torch.tensor ([5], dtype = torch.float32)

input_size = n_features
output_size = n_features

# model = nn.Linear (input_size, output_size)

class LinearRegression (nn.Module):
    def __init__(self, input_dim, output_dim):
        super (LinearRegression, self).__init__()
        self.lin = nn.Linear (input_dim, output_dim)

    def forward (self, X):
        return self.lin (X)
    
model = LinearRegression (input_dim=input_size, output_dim=output_size)

print (f'Prediction before training : f(5) = {model (X_test).item ():.3f}')

alpha = 0.01
epochs = 500

loss = nn.MSELoss ()
optimizer = torch.optim.SGD (model.parameters (), lr = alpha)

for epoch in range (epochs):
    y_pred = model (X)
    l = loss (y, y_pred)
    l.backward ()
    optimizer.step ()
    optimizer.zero_grad ()

    if epoch % 50 == 0:
        [w, b] = model.parameters ()
        print (f'Epoch {epoch + 50} : w = {w[0][0].item ()}, loss = {l:.8f}')

print (f'Prediction after training : f(5) = {model (X_test).item ():.3f}')