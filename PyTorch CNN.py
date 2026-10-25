import copy
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 42
BATCH_SIZE = 32
ARCH_EPOCHS = 10
IMPROVED_EPOCHS = 20
LEARNING_RATE = 0.001

DATA_DIR = "./data"
SAVE_DIR = Path("cnn_task_results")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_datasets():

    basic = transforms.ToTensor()

    augmented = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    normalized = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    base_data = datasets.CIFAR10(
        DATA_DIR,
        train=True,
        download=True,
        transform=basic
    )

    aug_data = datasets.CIFAR10(
        DATA_DIR,
        train=True,
        download=True,
        transform=augmented
    )

    norm_data = datasets.CIFAR10(
        DATA_DIR,
        train=True,
        download=True,
        transform=normalized
    )

    indices = torch.randperm(
        50000,
        generator=torch.Generator().manual_seed(SEED)
    ).tolist()

    train_ids = indices[:45000]
    val_ids = indices[45000:]

    baseline_sets = (
        Subset(base_data, train_ids),
        Subset(base_data, val_ids)
    )

    improved_sets = (
        Subset(aug_data, train_ids),
        Subset(norm_data, val_ids)
    )

    return (
        baseline_sets,
        improved_sets,
        base_data.classes,
        basic,
        normalized
    )

def get_loaders(train_set, val_set):

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader

class CNN(nn.Module):

    def __init__(
        self,
        filters=(32, 64, 64),
        kernels=(3, 3, 3),
        dense=(64,)
    ):

        super().__init__()

        blocks = []
        in_channels = 3

        for i, (out_channels, k) in enumerate(
            zip(filters, kernels)
        ):

            blocks.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=k
                )
            )

            blocks.append(nn.ReLU())

            if i in [0, 1]:
                blocks.append(
                    nn.MaxPool2d(
                        kernel_size=2,
                        stride=2
                    )
                )

            in_channels = out_channels

        self.features = nn.Sequential(*blocks)

        with torch.no_grad():

            dummy_image = torch.zeros(
                1, 3, 32, 32
            )

            feature_output = self.features(
                dummy_image
            )

            flat_size = feature_output.flatten(
                1
            ).shape[1]

        head = [nn.Flatten()]

        input_size = flat_size

        for neurons in dense:

            head.append(
                nn.Linear(input_size, neurons)
            )

            head.append(nn.ReLU())

            input_size = neurons

        head.append(nn.Linear(input_size, 10))

        self.classifier = nn.Sequential(*head)

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

class ImprovedCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3, 32, kernel_size=3,
                padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64, kernel_size=3,
                padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128, kernel_size=3,
                padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 4 * 4, 128),

            nn.ReLU(),

            nn.Dropout(p=0.4),

            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

EXPERIMENTS = {

    "A_original": dict(
        filters=(32, 64, 64),
        kernels=(3, 3, 3),
        dense=(64,)
    ),

    "B_kernel_5x5": dict(
        filters=(32, 64, 64),
        kernels=(5, 3, 3),
        dense=(64,)
    ),

    "C_four_convs": dict(
        filters=(32, 64, 64, 64),
        kernels=(3, 3, 3, 3),
        dense=(64,)
    ),

    "D_more_filters": dict(
        filters=(64, 128, 128),
        kernels=(3, 3, 3),
        dense=(64,)
    ),

    "E_more_dense": dict(
        filters=(32, 64, 64),
        kernels=(3, 3, 3),
        dense=(128, 64)
    )
}

def make_model(name):

    if name == "F_improved":
        return ImprovedCNN()

    return CNN(**EXPERIMENTS[name])

def one_epoch(
    model,
    loader,
    criterion,
    optimizer=None
):

    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            logits = model(images)

            loss = criterion(logits, labels)

            if training:

                loss.backward()

                optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        predictions = logits.argmax(dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_images += batch_size

    average_loss = total_loss / total_images
    accuracy = total_correct / total_images

    return average_loss, accuracy

def train_experiment(
    name,
    train_set,
    val_set,
    max_epochs,
    improved=False
):

    set_seed()

    model = make_model(name).to(DEVICE)

    train_loader, val_loader = get_loaders(
        train_set,
        val_set
    )

    criterion = nn.CrossEntropyLoss()

    if improved:

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=0.0001
        )

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=2
            )
        )

    else:

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        scheduler = None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_loss = float("inf")
    best_weights = None
    best_epoch = 0
    best_metrics = None

    patience_counter = 0

    for epoch in range(1, max_epochs + 1):

        train_loss, train_acc = one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = one_epoch(
            model,
            val_loader,
            criterion
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"{name} | Epoch {epoch}/{max_epochs} | "
            f"Train: {train_acc:.2%} | "
            f"Val: {val_acc:.2%} | "
            f"Val loss: {val_loss:.4f}"
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_loss - 1e-5:

            best_loss = val_loss

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch

            best_metrics = (
                train_acc,
                val_acc,
                val_loss
            )

            patience_counter = 0

        else:
            patience_counter += 1

        if improved and patience_counter >= 5:

            print("Early stopping activated")
            break

    model.load_state_dict(best_weights)

    model_path = SAVE_DIR / f"{name}.pth"

    torch.save(
        model.state_dict(),
        model_path
    )

    result = {
        "model": name,
        "parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "best_epoch": best_epoch,
        "train_accuracy": best_metrics[0],
        "validation_accuracy": best_metrics[1],
        "validation_loss": best_metrics[2],
        "checkpoint": str(model_path)
    }

    return result, history

def save_comparison(results, histories):

    ranked = sorted(
        results,
        key=lambda r: r["validation_loss"]
    )

    print("\nMODEL COMPARISON")

    for r in ranked:

        print(
            f"{r['model']} | "
            f"Parameters: {r['parameters']} | "
            f"Accuracy: {r['validation_accuracy']:.2%} | "
            f"Loss: {r['validation_loss']:.4f}"
        )

    with open(
        SAVE_DIR / "comparison.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(ranked[0].keys())
        )

        writer.writeheader()
        writer.writerows(ranked)

    plt.figure(figsize=(11, 5))

    plt.bar(
        [r["model"] for r in results],
        [
            100 * r["validation_accuracy"]
            for r in results
        ]
    )

    plt.ylabel("Validation Accuracy (%)")
    plt.title("CNN Architecture Comparison")
    plt.xticks(rotation=25, ha="right")

    plt.tight_layout()

    plt.savefig(
        SAVE_DIR / "model_comparison.png",
        dpi=170
    )

    plt.close()

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4)
    )

    for name in ["A_original", "F_improved"]:

        h = histories[name]

        x = range(
            1,
            len(h["train_loss"]) + 1
        )

        axes[0].plot(
            x, h["train_acc"],
            label=f"{name}: Train"
        )

        axes[0].plot(
            x, h["val_acc"],
            linestyle="--",
            label=f"{name}: Val"
        )

        axes[1].plot(
            x, h["train_loss"],
            label=f"{name}: Train"
        )

        axes[1].plot(
            x, h["val_loss"],
            linestyle="--",
            label=f"{name}: Val"
        )

    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")

    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    for ax in axes:
        ax.legend(fontsize=8)

    plt.tight_layout()

    plt.savefig(
        SAVE_DIR / "learning_curves.png",
        dpi=170
    )

    plt.close()

    return ranked[0]

def final_test(
    winner,
    basic_transform,
    normalized_transform,
    class_names
):

    name = winner["model"]

    model = make_model(name).to(DEVICE)

    model.load_state_dict(
        torch.load(
            winner["checkpoint"],
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    if name == "F_improved":
        transform = normalized_transform
    else:
        transform = basic_transform

    test_set = datasets.CIFAR10(
        DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loss, test_acc = one_epoch(
        model,
        test_loader,
        nn.CrossEntropyLoss()
    )

    print("\nFINAL TEST RESULT")
    print("Selected model:", name)
    print(f"Test accuracy: {test_acc:.2%}")
    print(f"Test loss: {test_loss:.4f}")

    images, labels = next(iter(test_loader))

    with torch.no_grad():

        predictions = model(
            images[:5].to(DEVICE)
        ).argmax(dim=1).cpu()

    fig, axes = plt.subplots(
        1, 5, figsize=(13, 3)
    )

    for i in range(5):

        image = images[i].clone()

        if name == "F_improved":
            image = image * 0.5 + 0.5

        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0)

        predicted = predictions[i].item()
        actual = labels[i].item()

        axes[i].imshow(image)

        axes[i].set_title(
            f"Pred: {class_names[predicted]}\n"
            f"True: {class_names[actual]}",
            color=(
                "blue" if predicted == actual
                else "red"
            )
        )

        axes[i].axis("off")

    plt.tight_layout()

    plt.savefig(
        SAVE_DIR / "test_predictions.png",
        dpi=170
    )

    plt.close()

    with open(
        SAVE_DIR / "test_result.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Selected model: {name}\n"
            f"Test accuracy: {test_acc:.4%}\n"
            f"Test loss: {test_loss:.4f}\n"
        )

def main():

    set_seed()

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Using device:", DEVICE)

    (
        baseline_sets,
        improved_sets,
        classes,
        basic,
        normalized
    ) = get_datasets()

    results = []
    histories = {}

    for name in EXPERIMENTS:

        result, history = train_experiment(
            name,
            *baseline_sets,
            ARCH_EPOCHS
        )

        results.append(result)
        histories[name] = history

    result, history = train_experiment(
        "F_improved",
        *improved_sets,
        IMPROVED_EPOCHS,
        improved=True
    )

    results.append(result)
    histories["F_improved"] = history

    winner = save_comparison(
        results,
        histories
    )

    final_test(
        winner,
        basic,
        normalized,
        classes
    )

    print("\nResults saved in:", SAVE_DIR.resolve())

if __name__ == "__main__":
    main()
