import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import glob

from vision_dataset import Suite2pVisionDataset
from vision_model import ROIVisionModel

def main():
    # 1. Discover sessions
    data_dir = Path('/mnt/other_ubunthu/mnt/data')
    session_paths = []
    # Search for sessions (using a subset for training)
    for stat_path in data_dir.rglob('stat.npy'):
        sp = stat_path.parent
        iscell_path = sp / 'iscell_final.npy'
        if not iscell_path.exists():
            iscell_path = sp / 'iscell.npy'
        if (sp / 'ops.npy').exists() and iscell_path.exists():
            session_paths.append(sp)
            
    # Use just 4-5 sessions for embedding training
    session_paths = sorted(session_paths)[:5]
    if not session_paths:
        print("No valid sessions found!")
        return
        
    print(f"Training Vision Embedder on {len(session_paths)} sessions:")
    for sp in session_paths:
        print(f" - {sp}")

    # 2. Setup Dataset and Loaders
    dataset = Suite2pVisionDataset(session_paths, target_size=(48, 48), is_train=True)
    
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 3. Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ROIVisionModel(embedding_dim=64).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 4. Training Loop (Phase A)
    epochs = 5 # Small number just to teach basic features
    print("\nStarting Phase A: Training Classification Head")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val Acc: {100 * correct / total:.2f}%")

    # 5. Save model weights
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "vision_embedder_weights.pth"
    torch.save(model.state_dict(), out_path)
    print(f"\nPhase A Complete. Model saved to {out_path}")
    print("The model can now be used as a feature extractor (Phase B)!")

if __name__ == '__main__':
    main()
