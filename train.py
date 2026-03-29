import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from models.customnet import CustomNet
from dataset.loaders import get_loaders

def train():
    # Initialize WandB experiment 
    wandb.init(project="faimdl-lab3", config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 64
    })
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomNet(num_classes=200).to(device)
    train_loader, val_loader = get_loaders('./data/tiny-imagenet-200', config.batch_size)
    
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

        # Log metrics to WandB [cite: 42, 48]
        # (Insert your validation logic here to get val_acc)
        # wandb.log({"epoch": epoch, "loss": loss.item(), "val_acc": val_acc})
        
        # Save best model to checkpoints/ 
        # torch.save(model.state_dict(), 'checkpoints/best_model.pth')

train()