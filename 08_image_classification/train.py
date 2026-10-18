"""
Training and Evaluation Pipeline for CIFAR-10 Image Classification with PyTorch CNN.
Leverages Apple Silicon MPS (GPU) acceleration, Data Augmentation, and Cosine Annealing.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Import model definition from local package
sys.path.append(os.path.dirname(__file__))
from model import CIFAR10CNN, CIFAR10_CLASSES, CLASS_EMOJIS, get_device


def print_banner(text: str, char: str = "=", width: int = 75):
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def get_data_loaders(data_dir: str, batch_size: int = 64):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # Data Augmentation & Normalization
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, train_dataset.classes


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, class_correct, class_total


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch CNN on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate (default: 0.001)")

    args = parser.parse_args()

    print_banner("🖼️ TRAINING DEEP CNN ON CIFAR-10 DATASET (PyTorch) 🖼️")

    device = get_device()
    print(f"  • Hardware Accelerator : {device.type.upper()} ({'Apple Silicon Metal' if device.type == 'mps' else 'CPU'})")

    data_dir = os.path.join(os.path.dirname(__file__), "data", "cifar10")
    if not os.path.exists(os.path.join(data_dir, "train")):
        print(f"Error: Dataset not found at '{data_dir}'. Please run download first.")
        return

    train_loader, test_loader, classes = get_data_loaders(data_dir, batch_size=args.batch_size)
    print(f"  • Training Samples     : {len(train_loader.dataset):,} images ({len(train_loader)} batches)")
    print(f"  • Test Samples         : {len(test_loader.dataset):,} images ({len(test_loader)} batches)")

    model = CIFAR10CNN(num_classes=10).to(device)
    print(f"  • Model Parameters     : {model.count_parameters():,} trainable weights")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print_banner("1. TRAINING LOOP")
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<12} | {'Test Loss':<12} | {'Test Acc':<12} | {'Time'}")
    print("-" * 75)

    best_acc = 0.0
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "cifar10_cnn.pt")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        scheduler.step()
        train_loss_epoch = train_loss / train_total
        train_acc_epoch = train_correct / train_total

        test_loss_epoch, test_acc_epoch, class_corr, class_tot = evaluate_model(model, test_loader, criterion, device)
        elapsed = time.time() - start_time

        print(f"  {epoch:>2}/{args.epochs:<3} | {train_loss_epoch:>10.4f} | {train_acc_epoch * 100:>9.2f}% | {test_loss_epoch:>10.4f} | {test_acc_epoch * 100:>9.2f}% | {elapsed:>5.1f}s")

        if test_acc_epoch > best_acc:
            best_acc = test_acc_epoch
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "best_acc": best_acc,
                "epochs": epoch
            }, checkpoint_path)

    # 2. Per-Class Accuracy Evaluation
    print_banner("2. PER-CLASS TEST SET ACCURACY BREAKDOWN")
    print(f"{'Class Name':<16} | {'Icon':<5} | {'Correct / Total':<18} | {'Accuracy':<10} | {'Visual Gauge'}")
    print("-" * 75)
    for i in range(10):
        c_name = classes[i]
        emoji = CLASS_EMOJIS.get(c_name, "")
        acc_pct = (class_corr[i] / class_tot[i]) * 100 if class_tot[i] > 0 else 0.0
        bar = "█" * int(acc_pct / 5)
        print(f"  {c_name:<14} | {emoji:<5} | {class_corr[i]:>4} / {class_tot[i]:>4} images | {acc_pct:>8.2f}% |{bar}")

    print_banner("3. MODEL PERSISTENCE & READINESS")
    print(f"  Champion weights saved to: {checkpoint_path}")
    print(f"  Peak Held-Out Test Accuracy: {best_acc * 100:.2f}%")
    print("  Ready for image inference and interactive classification!\n")


if __name__ == "__main__":
    main()
