import collections

import matplotlib.pyplot as plt
import torch
from torch import nn

from common import MLP, device, evaluate, prepare_data


@torch.no_grad()
def confusion_matrix(model, loader, n_classes):
    model.eval()
    matrix = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for x, y in loader:
        predictions = model(x.to(device)).argmax(dim=-1).cpu()
        for target, prediction in zip(y, predictions):
            matrix[target, prediction] += 1
    return matrix


def main():
    data = prepare_data()
    print(
        f"train={len(data.train_set)} val={len(data.val_set)} "
        f"test={len(data.test_set)} classes={len(data.classes)} "
        f"features={data.in_features}"
    )

    model = MLP(data.in_features, len(data.classes)).to(device)
    model.load_state_dict(
        torch.load("model.pt", map_location=device, weights_only=True)
    )

    matrix = confusion_matrix(model, data.val_loader, len(data.classes))
    _, ax = plt.subplots(figsize=(6.5, 6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(data.classes)))
    ax.set_xticklabels(data.classes, rotation=45, ha="right")
    ax.set_yticks(range(len(data.classes)))
    ax.set_yticklabels(data.classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion (val)")
    for i in range(len(data.classes)):
        for j in range(len(data.classes)):
            ax.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                fontsize=7,
            )
    plt.tight_layout()
    plt.show()

    recall = matrix.diag() / matrix.sum(dim=1).clamp(min=1)
    for class_name, class_recall in sorted(
        zip(data.classes, recall.tolist()), key=lambda item: item[1]
    ):
        print(f"  recall  {class_name:<12} {class_recall:.2%}")

    test_loss, test_acc = evaluate(
        model, data.test_loader, nn.CrossEntropyLoss()
    )
    majority = max(
        collections.Counter(
            int(data.test_set[index][1]) for index in range(len(data.test_set))
        ).values()
    ) / len(data.test_set)
    print(f"majority-class baseline (test): {majority:.2%}")
    print(f"FINAL TEST  loss={test_loss:.3f}  accuracy={test_acc:.2%}")
    print("beat baseline:", "YES" if test_acc > majority else "no -- keep iterating")


if __name__ == "__main__":
    main()