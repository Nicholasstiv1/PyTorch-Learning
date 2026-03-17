import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


# self.weights = nn.Parameter(
#     torch.randn(1, requires_grad=True, dtype=torch.float)
# )
# self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
