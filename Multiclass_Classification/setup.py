import matplotlib.pyplot as plt
import torch
from data import random_split
from MulticlassModel import MulticlassModel
from torch import nn

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_0 = MulticlassModel(input_features=2, output_features=4, hidden_units=8).to(
    device
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

X_train, X_test, y_train, y_test = random_split()
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test)

y_pred_probs = torch.softmax(y_logits, dim=1)
y_preds = torch.argmax(y_pred_probs, dim=1)

print(y_preds)
