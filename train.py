import collections

import matplotlib.pyplot as plt
import torch
from torch import nn

from common import MLP, device, evaluate, prepare_data


def train(
    model,
    train_loader,
    val_loader,
    *,
    epochs=10,
    lr=1e-3,
    weight_decay=0.0,
    loss_fn=None,
    max_grad_norm=None,
):
    model = model.to(device)
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, loss_fn)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(
            f"epoch {epoch + 1:2d}/{epochs}  "
            f"train_loss={train_loss:.3f} acc={train_acc:.2%}  |  "
            f"val_loss={val_loss:.3f} acc={val_acc:.2%}"
        )
    return history


def plot_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], "-o", label="train")
    ax1.plot(epochs, history["val_loss"], "-o", label="val")
    ax1.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["train_acc"], "-o", label="train")
    ax2.plot(epochs, history["val_acc"], "-o", label="val")
    ax2.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    data = prepare_data()
    print(
        f"train={len(data.train_set)} val={len(data.val_set)} "
        f"test={len(data.test_set)} classes={len(data.classes)} "
        f"features={data.in_features}"
    )

    counts = collections.Counter(
        int(data.train_set[index][1]) for index in range(len(data.train_set))
    )
    plt.figure(figsize=(9, 3))
    plt.bar(
        [data.classes[index] for index in sorted(counts)],
        [counts[index] for index in sorted(counts)],
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("count")
    plt.title("Training-set class balance")
    plt.tight_layout()
    plt.show()
    print(
        "majority-class baseline (train):",
        f"{max(counts.values()) / sum(counts.values()):.1%}",
    )

    model = MLP(data.in_features, len(data.classes))
    print(model)
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    train(
        model,
        data.train_loader,
        data.val_loader,
        epochs=8,
        lr=1e-3,
        weight_decay=1e-4,
    )
    torch.save(model.state_dict(), "model.pt")
    print("Saved model weights to model.pt")


if __name__ == "__main__":
    main()