# From https://saturncloud.io/blog/how-to-install-pytorch-on-the-gpu-with-docker/

import torch
import torch.nn as nn

device = ("cuda"
          if torch.cuda.is_available()
          else "mps"
          if torch.backends.mps.is_available()
          else "cpu"
          )

print("Device selected: %s" % device)
if device == "cuda":
    print("CUDA device selected: %s" % torch.cuda.current_device())
    print("CUDA device count %s" % torch.cuda.device_count())
    print("CUDA device name %s" % torch.cuda.get_device_name(torch.cuda.current_device()))


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)

print("Predicted class (TEST): %s" % y_pred)
