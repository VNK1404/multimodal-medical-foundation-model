# data_preprocessing.py
"""Utility script for loading, cleaning, splitting, and preprocessing the CheXpert
image dataset used in the BiomedCLIP fine‑tuning notebook.

The script follows exactly the steps that were performed inside the notebook, so
you can import the helper functions or run the file as a standalone module.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration – adjust these paths if your folder structure differs
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"d:/proejects/Multimodal Medical Foundation Model/image_only")
DATA_ROOT    = PROJECT_ROOT / "chexpert"
CSV_PATH     = DATA_ROOT / "train.csv"

# Target columns – must match the columns used elsewhere in the notebook
TARGETS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# ---------------------------------------------------------------------------
# 1️⃣ Load CSV and keep only needed columns
# ---------------------------------------------------------------------------
def load_raw_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only the path column (named "Path" in the original manifest) and the targets
    cols = ["Path"] + TARGETS
    df = df[cols]
    return df

# ---------------------------------------------------------------------------
# 2️⃣ Clean up paths – remove duplicated slashes and make them absolute
# ---------------------------------------------------------------------------
def fix_paths(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure single slash separators (some manifests contain "train/train/" etc.)
    df["Path"] = df["Path"].str.replace(r"/{2,}", "/", regex=True)
    # Prepend the absolute root directory so PIL can open the file directly
    df["Path"] = (DATA_ROOT / df["Path"]).astype(str)
    return df

# ---------------------------------------------------------------------------
# 3️⃣ Replace unknown labels (-1) with 0 (conservative) and cast to int
# ---------------------------------------------------------------------------
def sanitize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df[TARGETS] = df[TARGETS].replace(-1, 0).astype(int)
    return df

# ---------------------------------------------------------------------------
# 4️⃣ Drop rows whose image file does not exist (prevents runtime errors)
# ---------------------------------------------------------------------------
def drop_missing_images(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["Path"].apply(os.path.isfile)
    return df[mask].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 5️⃣ Stratified multi‑label split (keeps label distribution balanced)
# ---------------------------------------------------------------------------
def stratified_split(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # Take the first split as train/val
    train_idx, val_idx = next(mskf.split(df[TARGETS], df[TARGETS]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df

# ---------------------------------------------------------------------------
# 6️⃣ Compute class‑wise positive‑to‑negative ratios (used for FocalLoss weighting)
# ---------------------------------------------------------------------------
def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    pos_counts = train_df[TARGETS].sum()
    neg_counts = len(train_df) - pos_counts
    weights = (neg_counts / pos_counts.clip(lower=1)).values
    return torch.tensor(weights, dtype=torch.float32)

# ---------------------------------------------------------------------------
# 7️⃣ Dataset class – returns (image_tensor, label_tensor)
# ---------------------------------------------------------------------------
class RobustCheXpertDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["Path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row[TARGETS].values, dtype=torch.float32)
        return img, label

# ---------------------------------------------------------------------------
# 8️⃣ Default transforms (identical to those used in the notebook)
# ---------------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ---------------------------------------------------------------------------
# 9️⃣ Helper to build DataLoaders (batch size tuned for a single T4 GPU)
# ---------------------------------------------------------------------------
def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 128,
    num_workers: int = 6,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = RobustCheXpertDataset(train_df, transform=train_transform)
    val_dataset   = RobustCheXpertDataset(val_df,   transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # double batch for inference (no grads)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    return train_loader, val_loader

# ---------------------------------------------------------------------------
# 10️⃣ Example usage (you can comment this out if you only import the module)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_raw   = load_raw_dataframe(CSV_PATH)
    df_fixed = fix_paths(df_raw)
    df_clean = sanitize_labels(df_fixed)
    df_ok    = drop_missing_images(df_clean)
    train_df, val_df = stratified_split(df_ok, seed=42)
    class_weights = compute_class_weights(train_df)
    print("Class weights (neg/pos ratio):")
    for t, w in zip(TARGETS, class_weights.tolist()):
        print(f"  {t:<20}: {w:.2f}")
    train_loader, val_loader = build_dataloaders(train_df, val_df)
    print(f"✅ Train loader  → {len(train_loader)} batches, {len(train_loader.dataset):,} samples")
    print(f"✅ Val loader    → {len(val_loader)} batches, {len(val_loader.dataset):,} samples")
