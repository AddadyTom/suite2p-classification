import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from pathlib import Path

class Suite2pVisionDataset(Dataset):
    def __init__(self, session_paths, target_size=(48, 48), is_train=True):
        """
        Args:
            session_paths (list of str/Path): Paths to suite2p session directories.
            target_size (tuple): Output size of the image crops (H, W).
            is_train (bool): If True, loads labels. If False, labels are ignored/None.
        """
        self.session_paths = [Path(p) for p in session_paths]
        self.target_size = target_size
        self.is_train = is_train
        
        self.samples = [] # List of tuples: (session_idx, roi_idx, med, radius, ypix, xpix, label)
        self.session_data = {} # Cache for ops and iscell
        
        # Load metadata and prepare samples
        for s_idx, sp in enumerate(self.session_paths):
            ops_path = sp / 'ops.npy'
            stat_path = sp / 'stat.npy'
            iscell_path = sp / 'iscell_final.npy'
            if not iscell_path.exists():
                iscell_path = sp / 'iscell.npy'
                
            if not ops_path.exists() or not stat_path.exists():
                print(f"Skipping {sp}: ops.npy or stat.npy missing.")
                continue
                
            ops = np.load(ops_path, allow_pickle=True).item()
            stat = np.load(stat_path, allow_pickle=True)
            
            labels = None
            if self.is_train and iscell_path.exists():
                labels = np.load(iscell_path)[:, 0].astype(np.float32)
                
            # Cache the images
            meanImg = ops.get('meanImg')
            max_proj = ops.get('max_proj')
            
            if meanImg is None:
                print(f"Skipping {sp}: meanImg not found in ops.")
                continue
                
            if max_proj is None:
                max_proj = meanImg # Fallback if max_proj is missing
                
            # Normalize images to [0, 1] for CNN
            def normalize_img(img):
                img_min, img_max = np.percentile(img, 1), np.percentile(img, 99)
                img = (img - img_min) / (img_max - img_min + 1e-6)
                return np.clip(img, 0, 1)
                
            self.session_data[s_idx] = {
                'meanImg': normalize_img(meanImg),
                'max_proj': normalize_img(max_proj),
                'Ly': ops['Ly'],
                'Lx': ops['Lx']
            }
            
            for i, s in enumerate(stat):
                label = labels[i] if labels is not None else -1.0
                radius = s.get('radius', 5.0)
                self.samples.append({
                    's_idx': s_idx,
                    'med': s['med'],
                    'radius': radius,
                    'ypix': s['ypix'],
                    'xpix': s['xpix'],
                    'label': label
                })
                
        print(f"Initialized Vision Dataset with {len(self.samples)} ROIs from {len(self.session_data)} sessions.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        s_idx = sample['s_idx']
        s_data = self.session_data[s_idx]
        
        med = sample['med']
        radius = sample['radius']
        Ly, Lx = s_data['Ly'], s_data['Lx']
        
        # Dynamic box size: 3 * radius (min 16 pixels)
        box_size = max(16, int(3 * radius))
        half_box = box_size // 2
        
        y_center, x_center = int(med[0]), int(med[1])
        
        y_min = max(0, y_center - half_box)
        y_max = min(Ly, y_center + half_box)
        x_min = max(0, x_center - half_box)
        x_max = min(Lx, x_center + half_box)
        
        # In case the ROI is exactly at the boundary, we pad the extracted region
        patch_mean = np.zeros((box_size, box_size), dtype=np.float32)
        patch_max = np.zeros((box_size, box_size), dtype=np.float32)
        patch_mask = np.zeros((box_size, box_size), dtype=np.float32)
        
        # Destination indices in the patch array
        py_start = half_box - (y_center - y_min)
        py_end = py_start + (y_max - y_min)
        px_start = half_box - (x_center - x_min)
        px_end = px_start + (x_max - x_min)
        
        # Copy image data
        patch_mean[py_start:py_end, px_start:px_end] = s_data['meanImg'][y_min:y_max, x_min:x_max]
        patch_max[py_start:py_end, px_start:px_end] = s_data['max_proj'][y_min:y_max, x_min:x_max]
        
        # Create mask
        # Only include pixels within the ROI
        valid_ypix = sample['ypix']
        valid_xpix = sample['xpix']
        
        # Shift mask to patch coordinates
        mask_y = valid_ypix - y_center + half_box
        mask_x = valid_xpix - x_center + half_box
        
        # Filter out pixels that fall outside the patch (should be rare with 3*radius, but just in case)
        valid_idx = (mask_y >= 0) & (mask_y < box_size) & (mask_x >= 0) & (mask_x < box_size)
        patch_mask[mask_y[valid_idx], mask_x[valid_idx]] = 1.0
        
        # Stack into (3, H, W) tensor
        img_tensor = torch.from_numpy(np.stack([patch_mean, patch_max, patch_mask]))
        
        # Resize to fixed target size (48x48)
        img_tensor = TF.resize(img_tensor, self.target_size, antialias=True)
        
        label = torch.tensor([sample['label']], dtype=torch.float32)
        
        return img_tensor, label
