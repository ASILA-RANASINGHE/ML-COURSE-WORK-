"""
train.py - training and validation loops for Exercise 1 (owner: Member C)

What this file gives the team:
    train_one_epoch()   one pass over the training data, updating the weights
    evaluate_epoch()    one pass over any loader WITHOUT learning (validation / test)
    train_model()       the full training run: loss, optimiser, history, best checkpoint
    save_history(), load_history()   history <-> JSON file
"""

import json
import time

import torch
import torch.nn as nn


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns (average loss, accuracy) over the training data."""
    model.train()                                   # training mode (matters for dropout in Ex 3)
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()                       # 1. clear gradients from the last batch
        outputs = model(images)                     # 2. forward pass -> logits [B, 10]
        loss = criterion(outputs, labels)           # 3. cross-entropy loss for this batch
        loss.backward()                             # 4. backpropagation: compute gradients
        optimizer.step()                            # 5. update the weights

        total_loss += loss.item() * labels.size(0)  # loss.item() is the batch MEAN, so re-weight
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


@torch.no_grad()                                    # no gradients: faster, less memory
def evaluate_epoch(model, loader, criterion, device):
    """Score the model on a loader without changing it. Returns (average loss, accuracy)."""
    model.eval()                                    # evaluation mode
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3,
                device="cpu", save_path="best_mlp.pt", verbose=True):
    """Full training run with CrossEntropyLoss + Adam.

    After every epoch the model is scored on the VALIDATION set. Whenever the
    validation loss reaches a new minimum, the weights are saved to save_path.
    The test set is never used here.

    Returns history = {"train_loss": [...], "train_acc": [...],
                       "val_loss": [...], "val_acc": [...],
                       "best_epoch": int, "train_time_s": float}
    (one list entry per epoch; accuracies are fractions 0-1; best_epoch counts from 1)
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()               # applies log-softmax to the logits itself
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss, best_epoch = float("inf"), 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:                # new best -> save these weights
            best_val_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), save_path)

        if verbose:
            mark = " <- best so far" if best_epoch == epoch else ""
            print(f"Epoch {epoch:2d}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} "
                  f"| val loss {val_loss:.4f} acc {val_acc:.4f}{mark}")

    history["best_epoch"] = best_epoch
    history["train_time_s"] = time.time() - start
    return history


def save_history(history, path):
    """Write the history dict to a JSON file (used for plots and the report)."""
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def load_history(path):
    """Read a history dict back from a JSON file."""
    with open(path) as f:
        return json.load(f)