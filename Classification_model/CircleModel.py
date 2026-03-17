import torch
from torch import nn

# device = "cuda" if torch.cuda.is_available() else "cpu"


class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(
            in_features=2, out_features=5
        )  # takes in 2 features and upscales to 5 features
        self.layer_2 = nn.Linear(
            in_features=5, out_features=1
        )  # takes in 5 features from previous layer and outputs a single feature (same shape as y)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))  # x -> layer_1 -> layer_2 -> output


# model_0 = CircleModel().to(device)
