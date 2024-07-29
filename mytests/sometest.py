import torch.nn.functional as f

import torch


y = torch.tensor([52.3, 48.5, 50.0, 49])
y_hat = torch.tensor([51.3, 49.5, 50.0, 45])

loss = 0.0

loss_first = f.mse_loss(y.view(-1, 1), y_hat.view(-1, 1), reduction='mean')

loss += loss_first

print(type(loss))
print(loss)
