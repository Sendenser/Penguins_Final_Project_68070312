import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
if device == "cpu":
    print("WARNING: no GPU. Runtime -> Change runtime type -> T4 GPU, then re-run.")


@dataclass
class DataBundle:
    train_set: TensorDataset
    val_set: TensorDataset
    test_set: TensorDataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    classes: list[str]
    in_features: int


def prepare_data() -> DataBundle:
    df = sns.load_dataset("penguins").dropna().reset_index(drop=True)
    target = "species"
    classes = sorted(df[target].unique())
    y = torch.tensor([classes.index(species) for species in df[target]])

    num_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    cat_cols = ["island", "sex"]
    features = pd.concat(
        [df[num_cols], pd.get_dummies(df[cat_cols], dtype=float)], axis=1
    )
    full = TensorDataset(torch.tensor(features.values, dtype=torch.float32), y)

    n_test = len(full) // 5
    n_val = len(full) // 5
    n_train = len(full) - n_val - n_test
    generator = torch.Generator().manual_seed(SEED)
    train_split, val_split, test_split = random_split(
        full, [n_train, n_val, n_test], generator=generator
    )

    scaler = StandardScaler().fit(
        full.tensors[0][train_split.indices, :len(num_cols)].numpy()
    )

    def scale(split):
        indices = split.indices
        features = full.tensors[0][indices].clone()
        features[:, :len(num_cols)] = torch.tensor(
            scaler.transform(features[:, :len(num_cols)].numpy()),
            dtype=torch.float32,
        )
        return TensorDataset(features, full.tensors[1][indices])

    train_set, val_set, test_set = (
        scale(train_split),
        scale(val_split),
        scale(test_split),
    )

    batch_size = 16
    return DataBundle(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        train_loader=DataLoader(train_set, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(val_set, batch_size=batch_size),
        test_loader=DataLoader(test_set, batch_size=batch_size),
        classes=classes,
        in_features=features.shape[1],
    )


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


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * len(y)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n
