from functools import cache
from pathlib import Path

from jaxtyping import Float
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .nn_module_tools import convert_to_buffer


class MNISTClassifier(nn.Module): 
    def __init__(self) -> None: 
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2) 
        self.dropout1 = nn.Dropout2d(0.25) 
        self.dropout2 = nn.Dropout2d(0.5) 
        self.fc1 = nn.Linear(64 * 7 * 7, 128) 
        self.fc2 = nn.Linear(128, 10) 
  
    def forward(
        self, 
        x: Float[Tensor, "batch 1 height width"]
    ) -> Float[Tensor, "batch 10"]: 
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.dropout1(x) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.dropout2(x) 
        x = x.view(-1, 64 * 7 * 7) 
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x) 
        return x


@cache
def get_classifier(
    model_path: str,
    device: torch.device,
) -> MNISTClassifier:
    classifier = MNISTClassifier()
    classifier.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True), 
        strict=False
    )
    convert_to_buffer(classifier, persistent=False)
    classifier.eval()
    return classifier.to(device)
