import argparse
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import onnx
import onnxruntime as ort
import numpy as np

from model import EyeCNN
from prepare_data import prepare_dataset

def get_safe_device():
    """Detects if CUDA is usable without crashing on unsupported compute capabilities."""
    if torch.cuda.is_available():
        try:
            # Perform a test CUDA calculation
            t = torch.zeros((2, 2), device="cuda")
            _ = t + 1
            device_name = torch.cuda.get_device_name(0)
            print(f"[Device] Using GPU: {device_name}")
            return torch.device("cuda")
        except Exception as e:
            print(f"[Device] CUDA device detected but kernel execution failed ({e}).")
            print("[Device] Falling back safely to high-speed CPU.")
    else:
        print("[Device] CUDA not available, using CPU.")
    return torch.device("cpu")

def export_to_onnx(model, onnx_path, input_shape=(1, 3, 64, 64)):
    """Exports PyTorch model to ONNX and verifies it."""
    print(f"\n[ONNX] Exporting trained model to {onnx_path}...")
    model.eval()
    dummy_input = torch.randn(*input_shape, device="cpu")
    
    # Send model copy to CPU for safe export
    cpu_model = EyeCNN(num_classes=2)
    cpu_model.load_state_dict(model.state_dict())
    cpu_model.eval()

    torch.onnx.export(
        cpu_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print(f"[ONNX] Exported successfully: {onnx_path}")

    # Validate with onnx checker
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[ONNX] Validation check passed (model is structurally valid).")

    # Test run with onnxruntime
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = session.run(None, ort_inputs)
    print(f"[ONNX] Test inference with ONNX Runtime passed! Output shape: {ort_outs[0].shape}")

def train():
    parser = argparse.ArgumentParser(description="Train EyeCNN on Kaggle eye dataset")
    parser.add_argument("--dataset", "-d", default="dataset",
                        help="Dataset directory (default: dataset)")
    parser.add_argument("--epochs", "-e", type=int, default=10,
                        help="Number of epochs to train (default: 10)")
    args = parser.parse_args()

    project_dir = Path("D:/AI_CNN")
    data_dir = project_dir / args.dataset
        
    print(f"[Dataset] Using Kaggle data from: {data_dir.resolve()}")
    
    # Step 1: Ensure dataset is ready
    if not (data_dir / "train").exists() or not (data_dir / "val").exists():
        prepare_dataset(dest_dir=str(data_dir))
    
    # Step 2: Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Step 3: Datasets & Dataloaders
    train_dataset = datasets.ImageFolder(root=str(data_dir / "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=str(data_dir / "val"), transform=val_transforms)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    print(f"[Dataset] Classes: {class_names}")
    print(f"[Dataset] Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Step 4: Model, Loss, Optimizer
    device = get_safe_device()
    model = EyeCNN(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    num_epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_weights_path = project_dir / "best_model.pth"
    onnx_path = project_dir / "eye_cnn.onnx"

    print("\n" + "="*60)
    print(f"Starting CNN Training for {num_epochs} Epochs...")
    print("="*60)

    start_total_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Training phase
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
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_epoch_loss = train_loss / train_total
        train_epoch_acc = (train_correct / train_total) * 100.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = (val_correct / val_total) * 100.0
        epoch_time = time.time() - epoch_start

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_time:.2f}s) | "
              f"LR: {current_lr:.5f} | "
              f"Train Loss: {train_epoch_loss:.4f}, Acc: {train_epoch_acc:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.2f}%")

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), best_weights_path)

    total_time = time.time() - start_total_time
    print("="*60)
    print(f"Training completed in {total_time:.2f} seconds!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Weights saved to: {best_weights_path}")
    print("="*60)

    # Load best weights before ONNX export
    model.load_state_dict(torch.load(best_weights_path, map_location="cpu"))
    export_to_onnx(model, str(onnx_path))

    print(f"\n[Completed] Everything done! ONNX model ready at: {onnx_path.resolve()}")

if __name__ == "__main__":
    train()
