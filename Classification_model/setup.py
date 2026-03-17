import torch
from CircleModel import CircleModel
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model_0 = CircleModel().to(device)

loss_fn = nn.BCEWithLogitsLoss()  # Sigmoid activation function built-in

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

