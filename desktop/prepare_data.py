import os
import shutil
import random
from pathlib import Path
import kagglehub

def prepare_dataset(dest_dir="D:/AI_CNN/dataset", train_ratio=0.8, seed=42):
    random.seed(seed)
    dest_path = Path(dest_dir)
    train_closed = dest_path / "train" / "closed"
    train_open = dest_path / "train" / "open"
    val_closed = dest_path / "val" / "closed"
    val_open = dest_path / "val" / "open"

    for d in [train_closed, train_open, val_closed, val_open]:
        d.mkdir(parents=True, exist_ok=True)

    # Check if already populated
    n_train = len(list(train_closed.glob("*.*"))) + len(list(train_open.glob("*.*")))
    if n_train > 0:
        print(f"[Info] Dataset already prepared in {dest_dir} ({n_train} training images found).")
        return dest_path

    print("[1/3] Locating / downloading dataset from Kaggle...")
    raw_path = kagglehub.dataset_download("dheerajperumandla/drowsiness-dataset")
    print(f"[Info] Raw Kaggle dataset path: {raw_path}")

    raw_dir = Path(raw_path)
    # Locate Closed and Open folders
    closed_sources = list(raw_dir.rglob("Closed"))
    open_sources = list(raw_dir.rglob("Open"))

    if not closed_sources or not open_sources:
        raise FileNotFoundError(f"Could not find 'Closed' and 'Open' folders in {raw_dir}")

    closed_dir = closed_sources[0]
    open_dir = open_sources[0]

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    closed_imgs = [p for p in closed_dir.iterdir() if p.suffix.lower() in valid_exts]
    open_imgs = [p for p in open_dir.iterdir() if p.suffix.lower() in valid_exts]

    print(f"[2/3] Found {len(closed_imgs)} closed eye images and {len(open_imgs)} open eye images.")

    random.shuffle(closed_imgs)
    random.shuffle(open_imgs)

    def split_and_copy(images, train_target, val_target):
        split_idx = int(len(images) * train_ratio)
        train_files = images[:split_idx]
        val_files = images[split_idx:]
        for f in train_files:
            shutil.copy2(f, train_target / f.name)
        for f in val_files:
            shutil.copy2(f, val_target / f.name)
        return len(train_files), len(val_files)

    print("[3/3] Copying and splitting dataset (80% train / 20% val)...")
    tc, vc = split_and_copy(closed_imgs, train_closed, val_closed)
    to, vo = split_and_copy(open_imgs, train_open, val_open)

    print(f"[Done] Dataset prepared successfully:")
    print(f"  - Train: {tc} closed, {to} open (Total: {tc + to})")
    print(f"  - Val:   {vc} closed, {vo} open (Total: {vc + vo})")
    print(f"  - Output directory: {dest_path.resolve()}")
    return dest_path

if __name__ == "__main__":
    prepare_dataset()
