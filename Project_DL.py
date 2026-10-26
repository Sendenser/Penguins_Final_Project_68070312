import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import collections
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

# --- reproducibility: fix the seed so your reported numbers reproduce -------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- device: train on the GPU if Colab gave you one -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
if device == "cpu":
    print("WARNING: no GPU. Runtime -> Change runtime type -> T4 GPU, then re-run.")


@torch.no_grad()
def accuracy(logits, y):
    """Fraction correct for a batch of class logits vs integer labels."""
    return (logits.argmax(dim=-1) == y).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    """One no-grad pass over `loader`; returns (mean loss, accuracy)."""
    model.eval()                                  # dropout off, BN uses running stats
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)         # same device as the model
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * len(y)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def train(model, train_loader, val_loader, *, epochs=10, lr=1e-3,
          weight_decay=0.0, loss_fn=None, max_grad_norm=None):
    """The five-step loop (forward, loss, zero_grad, backward, step) with a
    validation pass each epoch. Returns a per-epoch history dict."""
    model = model.to(device)                      # move the model once, before the loop
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:                 # the five steps, per batch
            x, y = x.to(device), y.to(device)     # data on the SAME device as the model
            logits = model(x)                     # 1. forward
            loss = loss_fn(logits, y)             # 2. loss
            optimizer.zero_grad()                 # 3. clear last batch's grads
            loss.backward()                       # 4. backprop
            if max_grad_norm is not None:         # clip for RNNs/LSTMs (Track B)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()                      # 5. update the weights

        tr_loss, tr_acc = evaluate(model, train_loader, loss_fn)
        va_loss, va_acc = evaluate(model, val_loader, loss_fn)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        print(f"epoch {epoch + 1:2d}/{epochs}  "
              f"train_loss={tr_loss:.3f} acc={tr_acc:.2%}  |  "
              f"val_loss={va_loss:.3f} acc={va_acc:.2%}")
    return history


def plot_curves(history):
    """Train vs. validation loss and accuracy — read the gap (labs L5/L11)."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], "-o", label="train")
    ax1.plot(epochs, history["val_loss"], "-o", label="val")
    ax1.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["train_acc"], "-o", label="train")
    ax2.plot(epochs, history["val_acc"], "-o", label="val")
    ax2.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("spine ready: accuracy(), evaluate(), train(), plot_curves()")

norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
train_tf = transforms.Compose([                 # augment ONLY the training set
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(), norm,
])
eval_tf = transforms.Compose([transforms.ToTensor(), norm])   # never augment val/test
# CIFAR's official split is 50k train + 10k test. We hold the 10k test out
# untouched, and carve a validation set off the training data.
full_train = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
test_set = datasets.CIFAR10("./data", train=False, download=True, transform=eval_tf)

# Subset to train fast on a T4 (~3-4 min). Use all 50k for a stronger result.
SUBSET = 10000
g = torch.Generator().manual_seed(SEED)
idx = torch.randperm(len(full_train), generator=g)[:SUBSET].tolist()
n_val = SUBSET // 5
train_idx, val_idx = idx[n_val:], idx[:n_val]
train_set = Subset(full_train, train_idx)
# val must NOT be augmented -> point it at a non-augmented copy of the data
val_base = datasets.CIFAR10("./data", train=True, download=True, transform=eval_tf)
val_set = Subset(val_base, val_idx)

batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)
classes = full_train.classes                    # ['airplane', 'automobile', ...]
print(f"train={len(train_set)} val={len(val_set)} test={len(test_set)} "
      f"classes={len(classes)}")

# ============================================================================
# TRACK C · TABULAR — Penguins (real measurements, 3 species)
# Build an MLP. UNCOMMENT this whole block to use it, and comment out Track A.
# ============================================================================

# Real/messy: a real ecology dataset with MISSING values (rows with NaN) and
# categorical columns (island, sex) you must encode. 344 rows, 3 species.
df = sns.load_dataset("penguins").dropna().reset_index(drop=True)
target = "species"
classes = sorted(df[target].unique())          # ['Adelie', 'Chinstrap', 'Gentoo']
y = torch.tensor([classes.index(s) for s in df[target]])

num_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
cat_cols = ["island", "sex"]
X_df = pd.concat([df[num_cols], pd.get_dummies(df[cat_cols], dtype=float)], axis=1)
in_features = X_df.shape[1]
n_classes = len(classes)

full = TensorDataset(torch.tensor(X_df.values, dtype=torch.float32), y)
n_test = len(full) // 5
n_val = len(full) // 5
n_train = len(full) - n_val - n_test
g = torch.Generator().manual_seed(SEED)
train_set, val_set, test_set = random_split(full, [n_train, n_val, n_test], generator=g)

# Standardize numeric features: FIT ON TRAIN ONLY, reuse on val/test (no leakage).
train_X = full.tensors[0][train_set.indices]
scaler = StandardScaler().fit(train_X[:, :len(num_cols)].numpy())
def scale(ds):
    xb = full.tensors[0][ds.indices].clone()
    xb[:, :len(num_cols)] = torch.tensor(
        scaler.transform(xb[:, :len(num_cols)].numpy()), dtype=torch.float32)
    return TensorDataset(xb, full.tensors[1][ds.indices])
train_set, val_set, test_set = scale(train_set), scale(val_set), scale(test_set)

batch_size = 16
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)
print(f"train={n_train} val={n_val} test={n_test} "
      f"classes={n_classes} features={in_features}")

# 1. Class histogram of the training split -- is it balanced?
counts = collections.Counter(int(train_set[i][1]) for i in range(len(train_set)))
plt.figure(figsize=(9, 3))
plt.bar([classes[k] for k in sorted(counts)], [counts[k] for k in sorted(counts)])
plt.xticks(rotation=45, ha="right"); plt.ylabel("count")
plt.title("Training-set class balance"); plt.tight_layout(); plt.show()
print("majority-class baseline (train):",
      f"{max(counts.values()) / sum(counts.values()):.1%}")

# ---- TRACK C · MLP -- a couple of hidden layers with dropout.
class MLP(nn.Module):
    def __init__(self, in_features, n_classes, hidden=32, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x):
        return self.net(x)
model = MLP(in_features, n_classes)
n_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"trainable parameters: {n_params:,}")

# 🔧 Your main run. Tune these on VALIDATION; do not touch test yet.
history = train(
    model,
    train_loader,
    val_loader,
    epochs=8,            # 🔧 enough to see val accuracy plateau
    lr=1e-3,             # 🔧 if loss diverges/NaN, drop 10x; if it crawls, raise it
    weight_decay=1e-4,   # 🔧 L2 regularization (Track B: also max_grad_norm=1.0)
)

@torch.no_grad()
def confusion_matrix(model, loader, n_classes):
    model.eval()
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for x, y in loader:
        preds = model(x.to(device)).argmax(dim=-1).cpu()
        for t, p in zip(y, preds):
            cm[t, p] += 1
    return cm

cm = confusion_matrix(model, val_loader, len(classes))
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion (val)")
for i in range(len(classes)):                    # annotate counts
    for j in range(len(classes)):
        ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
plt.tight_layout(); plt.show()

# Per-class recall (of the true class c, what fraction did we catch?)
recall = cm.diag() / cm.sum(dim=1).clamp(min=1)
for c, r in sorted(zip(classes, recall.tolist()), key=lambda t: t[1]):
    print(f"  recall  {c:<12} {r:.2%}")

# ✅ ONE-SHOT held-out test evaluation. Do NOT tune after seeing this number.
test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss())
majority = max(
    collections.Counter(int(test_set[i][1]) for i in range(len(test_set))).values()
) / len(test_set)
print(f"majority-class baseline (test): {majority:.2%}")
print(f"FINAL TEST  loss={test_loss:.3f}  accuracy={test_acc:.2%}")
print("beat baseline:", "YES" if test_acc > majority else "no -- keep iterating")