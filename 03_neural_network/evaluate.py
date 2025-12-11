import torch
from utils import plot_images

def evaluate_model(model, test_loader, device='cpu', num_samples=5):
    """
    Evaluate the model on test set and display a few sample predictions
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    images_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # collect some samples for visualization
            if len(images_list) < num_samples:
                images_list.extend(images[:num_samples].cpu())
                labels_list.extend(labels[:num_samples].cpu())
                preds_list.extend(predicted[:num_samples].cpu())

    accuracy = correct / total
    print(f"Test set accuracy: {accuracy*100:.2f}%")

    # Show sample predictions
    plot_images(images_list[:num_samples], labels_list[:num_samples], preds_list[:num_samples])
