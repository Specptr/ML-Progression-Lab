# 2025 12 11
import torch
from model import MNISTClassifier
from utils import get_data_loaders
from train import train_model
from evaluate import evaluate_model

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # 1. Load data
    train_loader, test_loader = get_data_loaders(batch_size=64)

    # 2. Initialize model
    model = MNISTClassifier()

    # 3. Train model
    model = train_model(
        model,
        train_loader,
        epochs=10,
        lr=0.001,
        device=device
    )

    # 4. Evaluate model
    evaluate_model(
        model,
        test_loader,
        device=device,
        num_samples=5
    )

if __name__ == "__main__":
    main()