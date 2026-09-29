import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from pathlib import Path

# Set random seed for exact reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MiniRocketExtractor:
    """
    Ultra-lightweight, memory-safe MiniRocket feature extractor.
    Generates fixed random multi-scale 1D convolutional kernels of length 9,
    with weights in {-1, 2}, dilations d in {1, 2, 4, 8, 16, 32, 64, 128},
    and computes PPV (Proportion of Positive Values) + Max peak response.
    """
    def __init__(self, num_kernels=84, max_dilations=8, seed=42):
        self.num_kernels = num_kernels
        self.max_dilations = max_dilations
        self.seed = seed
        self.dilations = [2**i for i in range(max_dilations)] # [1, 2, 4, 8, 16, 32, 64, 128]
        self.weights = None
        self.biases = None
        self._init_kernels()

    def _init_kernels(self):
        rng = np.random.RandomState(self.seed)
        
        # MiniRocket length-9 patterns with 3 positive weights (+2) and 6 negative weights (-1)
        # Sum of weights is 3*2 + 6*(-1) = 0 (mean zero filters)
        base_patterns = []
        for _ in range(self.num_kernels):
            kernel = np.full(9, -1.0, dtype=np.float32)
            pos_idx = rng.choice(9, size=3, replace=False)
            kernel[pos_idx] = 2.0
            base_patterns.append(kernel)
        self.base_patterns = np.stack(base_patterns) # Shape: (num_kernels, 9)

    def fit_biases(self, sample_traces):
        """
        Fit random bias thresholds from quantiles of convolutions on a small subsample of traces.
        sample_traces shape: (N_sample, T)
        """
        device = torch.device("cpu")
        N_sample, T = sample_traces.shape
        x_tensor = torch.tensor(sample_traces[:, None, :], dtype=torch.float32, device=device) # (N_sample, 1, T)
        
        all_biases = []
        for d in self.dilations:
            # Dilated weights: (num_kernels, 1, 9)
            w_tensor = torch.tensor(self.base_patterns[:, None, :], dtype=torch.float32, device=device)
            # Convolution
            conv_out = F.conv1d(x_tensor, w_tensor, dilation=d, padding=d * 4) # (N_sample, num_kernels, T_out)
            conv_np = conv_out.cpu().numpy() # (N_sample, num_kernels, T_out)
            
            # Select random quantiles for biases
            biases_per_kernel = []
            for k in range(self.num_kernels):
                vals = conv_np[:, k, :].ravel()
                # Pick a random quantile between 0.1 and 0.9
                q = np.random.uniform(0.1, 0.9)
                bias = np.quantile(vals, q)
                biases_per_kernel.append(bias)
            all_biases.append(np.array(biases_per_kernel, dtype=np.float32))
            
        self.biases = np.stack(all_biases) # Shape: (len(dilations), num_kernels)
        print(f"Initialized MiniRocket kernels across {len(self.dilations)} dilations ({len(self.dilations) * self.num_kernels * 2} total features).")

    def transform(self, traces, batch_size=128, downsample_factor=4):
        """
        Transform traces of shape (N_cells, T) into MiniRocket embeddings of shape (N_cells, num_features).
        Uses downsampling (default 4x) and batching to keep memory under 50 MB.
        """
        N_cells, T = traces.shape
        if downsample_factor > 1:
            # Downsample trace by averaging or subsampling
            traces_ds = traces[:, ::downsample_factor].astype(np.float32)
        else:
            traces_ds = traces.astype(np.float32)
            
        T_ds = traces_ds.shape[1]
        device = torch.device("cpu")
        
        all_features = []
        
        # Process in memory-safe batches
        for start_idx in range(0, N_cells, batch_size):
            end_idx = min(start_idx + batch_size, N_cells)
            batch = traces_ds[start_idx:end_idx] # (B, T_ds)
            B = batch.shape[0]
            
            x_batch = torch.tensor(batch[:, None, :], dtype=torch.float32, device=device) # (B, 1, T_ds)
            batch_feats = []
            
            for d_idx, d in enumerate(self.dilations):
                # Dilated weights: (num_kernels, 1, 9)
                w_tensor = torch.tensor(self.base_patterns[:, None, :], dtype=torch.float32, device=device)
                
                # Fast 1D Convolution
                conv_out = F.conv1d(x_batch, w_tensor, dilation=d, padding=d * 4) # (B, num_kernels, T_out)
                
                # Apply bias if fitted
                if self.biases is not None:
                    b_tensor = torch.tensor(self.biases[d_idx, :, None], dtype=torch.float32, device=device) # (num_kernels, 1)
                    conv_biased = conv_out - b_tensor # (B, num_kernels, T_out)
                else:
                    conv_biased = conv_out
                    
                # 1. PPV: Proportion of Positive Values
                ppv = torch.mean((conv_biased > 0).float(), dim=-1) # (B, num_kernels)
                
                # 2. Max value
                max_val = torch.max(conv_out, dim=-1).values # (B, num_kernels)
                
                batch_feats.append(ppv.cpu().numpy())
                batch_feats.append(max_val.cpu().numpy())
                
            batch_feats = np.hstack(batch_feats) # Shape: (B, num_features)
            all_features.append(batch_feats)
            
        all_features = np.vstack(all_features) # Shape: (N_cells, num_features)
        return all_features

def extract_session_minirocket(raw_path, extractor, downsample_factor=4):
    """
    Load raw F and Fneu for a single session, compute scale-normalized F_corr,
    and extract MiniRocket embeddings with zero memory leaks.
    """
    raw_path = Path(raw_path)
    F = np.load(raw_path / 'F.npy', mmap_mode='r')
    Fneu = np.load(raw_path / 'Fneu.npy', mmap_mode='r')
    
    n_cells = F.shape[0]
    
    # Calculate session noise scale
    sample_size = min(200, n_cells)
    sample_F = np.array(F[:sample_size])
    sample_Fneu = np.array(Fneu[:sample_size])
    sample_Fc = sample_F - 0.7 * sample_Fneu
    scale = np.median(np.std(sample_Fc, axis=1))
    if np.isnan(scale) or scale <= 0:
        scale = 1.0
        
    # Process cells in chunks to keep memory usage minimal
    chunk_size = 256
    sess_embeddings = []
    
    for start_i in range(0, n_cells, chunk_size):
        end_i = min(start_i + chunk_size, n_cells)
        f_chunk = np.array(F[start_i:end_i], dtype=np.float32)
        fneu_chunk = np.array(Fneu[start_i:end_i], dtype=np.float32)
        
        fc_chunk = (f_chunk - 0.7 * fneu_chunk) / scale
        
        # Extract features
        feats = extractor.transform(fc_chunk, batch_size=128, downsample_factor=downsample_factor)
        sess_embeddings.append(feats)
        
        del f_chunk, fneu_chunk, fc_chunk
        
    sess_embeddings = np.vstack(sess_embeddings)
    gc.collect()
    return sess_embeddings
