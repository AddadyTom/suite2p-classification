import os
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from vision_dataset import Suite2pVisionDataset
from vision_model import get_vision_model
from apply_AI import extract_features, FEATURE_NAMES_25

def process_sessions_vision(source_dir, output_dir, model_path):
    source_path = Path(source_dir)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    print(f"Scanning for valid Suite2p sessions under: {source_path}")
    valid_sessions = []
    for stat_path in source_path.rglob('stat.npy'):
        session_dir = stat_path.parent
        if (session_dir / 'F.npy').exists() and (session_dir / 'ops.npy').exists():
            iscell_path = session_dir / 'iscell_final.npy'
            if not iscell_path.exists():
                iscell_path = session_dir / 'iscell.npy'
            if iscell_path.exists():
                valid_sessions.append(session_dir)
                
    valid_sessions = sorted(list(set(valid_sessions)))
    if not valid_sessions:
        print("No valid sessions found!")
        return
        
    print(f"Found {len(valid_sessions)} sessions.")
    
    # 1. Setup Vision Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vision_model = get_vision_model(embedding_dim=64, pretrained_path=model_path, device=device)
    vision_model.eval()
    
    # 2. Iterate through sessions
    X_all = []
    y_all = []
    groups_all = []
    
    for group_id, session_path in enumerate(valid_sessions):
        print(f"Processing session {group_id+1}/{len(valid_sessions)}: {session_path}")
        
        # A. Load Ground Truth
        iscell_path = session_path / 'iscell_final.npy'
        if not iscell_path.exists():
            iscell_path = session_path / 'iscell.npy'
        y = np.load(iscell_path)[:, 0].astype(int)
        
        # B. Extract Numerical Features (25 features)
        F = np.load(session_path / 'F.npy', mmap_mode='r')
        Fneu = np.load(session_path / 'Fneu.npy', mmap_mode='r')
        stat = np.load(session_path / 'stat.npy', allow_pickle=True)
        X_num = extract_features(F, Fneu, stat, FEATURE_NAMES_25)
        
        # C. Extract Vision Embeddings
        # We create a dataset just for this session
        dataset = Suite2pVisionDataset([session_path], target_size=(48, 48), is_train=False)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        
        embeddings_list = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                # Forward features only!
                embeds = vision_model.forward_features(imgs)
                embeddings_list.append(embeds.cpu().numpy())
                
        X_vis = np.vstack(embeddings_list)
        
        # Ensure sizes match
        assert len(y) == X_num.shape[0] == X_vis.shape[0], "Mismatch in ROI counts!"
        
        # Concatenate numerical and vision features (25 + 64 = 89)
        X_combined = np.hstack([X_num, X_vis])
        
        X_all.append(X_combined)
        y_all.append(y)
        groups_all.append(np.full(len(y), group_id))
        
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    groups = np.concatenate(groups_all)
    
    print(f"\nSuccessfully compiled dataset:")
    print(f"  Total ROIs: {X.shape[0]}")
    print(f"  Features:   {X.shape[1]}")
    
    np.save(out_path / 'X_vision_dataset.npy', X)
    np.save(out_path / 'y_vision_dataset.npy', y)
    np.save(out_path / 'groups_vision_dataset.npy', groups)
    print(f"Saved to: {out_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="/mnt/other_ubunthu/mnt/data")
    parser.add_argument('--output', type=str, default=".")
    parser.add_argument('--model', type=str, default="models/vision_embedder_weights.pth")
    args = parser.parse_args()
    
    process_sessions_vision(args.source, args.output, args.model)
