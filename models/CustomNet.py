import torch
from torch import nn
import torch.nn.functional as F

class CustomNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomNet, self).__init__()
        # Define some layers
        self.conv1 = nn.Conv2d(3,16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.25)    

    def forward(self, x):
        # Layer 1: Conv -> ReLU -> Pool
        # [B, 3, 64, 64] -> [B, 16, 64, 64] -> [B, 16, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))

        # Layer 2: Conv -> ReLU -> Pool
        # [B, 16, 32, 32] -> [B, 32, 32, 32] -> [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the "cube" into a "vector"
        # x.size(0) is the Batch size; -1 tells PyTorch to figure out the rest
        x = x.view(x.size(0), -1) 
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Regularization
        x = self.fc2(x)     # Raw scores (logits) for 200 classes
        
        return x 
