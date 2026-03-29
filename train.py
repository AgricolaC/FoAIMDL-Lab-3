import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from models.CustomNet import CustomNet
from data.tiny_imagenet import get_loaders
import os
from tqdm import tqdm



def train():
    # Initialize WandB experiment 
    wandb.init(project="faimdl-lab3", config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 64
    })
    config = wandb.config

    os.makedirs('checkpoints', exist_ok=True) # Ensure the folder exists!

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomNet(num_classes=200).to(device)
    train_loader, val_loader = get_loaders('./dataset/tiny-imagenet-200', config.batch_size)    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)

    # Optional: Watch model gradients 
    wandb.watch(model)

    best_acc = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        val_acc = validate(model, val_loader, criterion) 
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"--> New best model saved with {best_acc:.2f}% accuracy!")

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct, total = 0,0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass ONLY
            outputs = model(inputs)
            # Calculate loss
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()    
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total
    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy

train()