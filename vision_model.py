import torch
import torch.nn as nn
import torch.nn.functional as F

class ROIVisionModel(nn.Module):
    def __init__(self, embedding_dim=64):
        super(ROIVisionModel, self).__init__()
        
        # Input is 3 channels: meanImg, max_proj, mask
        # Image size is resized to 48x48
        
        # Block 1: 48x48 -> 24x24
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: 24x24 -> 12x12
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3: 12x12 -> 6x6
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Flatten and embed
        # 6 * 6 * 64 = 2304
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.drop = nn.Dropout(0.3)
        self.fc_embed = nn.Linear(128, embedding_dim)
        
        # Final classification head (will be ignored during feature extraction)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward_features(self, x):
        """Returns the embeddings (useful after training)"""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        embed = F.relu(self.fc_embed(x))
        return embed

    def forward(self, x):
        """Standard forward pass for training the binary classifier"""
        embed = self.forward_features(x)
        logits = self.classifier(embed)
        return logits

def get_vision_model(embedding_dim=64, pretrained_path=None, device='cpu'):
    model = ROIVisionModel(embedding_dim=embedding_dim)
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.to(device)
    return model
