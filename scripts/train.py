import torch
from utils.data_loader import get_loader
from models.vit_model import load_vit_model
from utils.plotter import plot_accuracy

from torch import nn
import numpy as np
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10):
    results = {"train_loss": [], "train_acc": [], "val_acc": []}

    model.to(device)
    model.train()

    for param in model.parameters():
        param.requires_grad = True

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_loss = train_loss / len(train_loader)
        results["train_loss"].append(avg_loss)
        results["train_acc"].append(train_acc)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, preds = torch.max(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        results["val_acc"].append(val_acc)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return results

if __name__ == "__main__":
    # Path setup
    pretrained_path = "checkpoint/ViT-B_16.npz"
    train_dir = "/content/drive/MyDrive/hymenoptera_data/train"
    val_dir = "/content/drive/MyDrive/hymenoptera_data/val"

    train_loader, val_loader = get_loader(train_dir, val_dir, img_size=224, batch_size=64)
    model = load_vit_model(pretrained_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    results = train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=18)

    plot_accuracy(results)

    torch.save(model.state_dict(), "checkpoints/vit_best_model.pth")
