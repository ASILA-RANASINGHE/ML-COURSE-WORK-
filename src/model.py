"""
model.py - Multi-Layer Perceptron for CIFAR-10, Exercise 1 (owner: Member B)

What this file gives the team:
    MLP                 the classifier (exactly three trainable nn.Linear layers)
    count_parameters()  number of trainable parameters (needed for the Ex 1 vs Ex 2 comparison)
    parameter_table()   per-layer breakdown, used in the report
"""

import torch.nn as nn


class MLP(nn.Module):
    """Input [B, 3, 32, 32] -> Flatten -> fc1 -> ReLU -> fc2 -> ReLU -> fc3 -> logits [B, 10].

    Trainable layers: fc1, fc2, fc3 (three nn.Linear layers).
    Flatten and ReLU have no weights, so they are not counted as trainable layers.
    forward() returns raw logits with NO softmax, because nn.CrossEntropyLoss
    applies log-softmax internally.
    """

    def __init__(self, hidden1=512, hidden2=256, num_classes=10, input_dim=3 * 32 * 32):
        super().__init__()
        self.flatten = nn.Flatten()                     # [B, 3, 32, 32] -> [B, 3072]
        self.fc1 = nn.Linear(input_dim, hidden1)        # trainable layer 1
        self.fc2 = nn.Linear(hidden1, hidden2)          # trainable layer 2
        self.fc3 = nn.Linear(hidden2, num_classes)      # trainable layer 3 (output)
        self.relu = nn.ReLU()                           # activation between layers

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)                              # logits, one score per class


def count_parameters(model):
    """Total number of trainable parameters (weights + biases)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_table(model):
    """List of (name, shape, count) for every trainable tensor, e.g. ('fc1.weight', (512, 3072), 1572864)."""
    return [(name, tuple(p.shape), p.numel())
            for name, p in model.named_parameters() if p.requires_grad]
