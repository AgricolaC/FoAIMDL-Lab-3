import torch
import torch.nn as nn
from models.CustomNet import CustomNet
from data.tiny_imagenet import get_loaders

def evaluate():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Recreate the Model Architecture
    model = CustomNet(num_classes=200).to(device)
    
    # 3. Load the Saved Weights
    # This pulls from the 'checkpoints' folder where you saved your progress [cite: 62]
    checkpoint_path = './checkpoints/best_model.pth'
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Successfully loaded weights from {checkpoint_path}")
    except FileNotFoundError:
        print("Error: No checkpoint found. Did you run train.py first?")
        return

    # 4. Get the Data (Val/Test loader only)
    _, val_loader = get_loaders('./dataset/tiny-imagenet-200', batch_size=64)
    
    criterion = nn.CrossEntropyLoss()
    
    # 5. Evaluation Loop 
    model.eval() # Set model to evaluation mode
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad(): # Disable gradient tracking to save memory/time
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    print("-" * 30)
    print(f'FINAL EVALUATION RESULTS')
    print(f'Loss: {avg_loss:.6f} | Accuracy: {accuracy:.2f}%')
    print("-" * 30)

if __name__ == "__main__":
    evaluate()