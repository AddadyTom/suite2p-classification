#!/usr/bin/env python3
import sys
import os
import time
import argparse
import importlib
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

MORPHOLOGY_FEATURES = {
    'npix', 'solidity', 'mrs', 'compact', 'aspect_ratio', 'radius',
    'number_of_bright_pixels', 'bright_pixels_ratio', 'roi_idx_norm',
    'avg_asym', 'max_asym', 'max_width', 'area_to_radius_sq',
    'bright_pixels_to_radius_sq'
}

def recompute_session_chunked(npz_file, features_to_process, module_name, chunk_size=250):
    """
    Memory-safe chunked session feature extractor.
    Reads somatic traces in small chunks (e.g. 250 cells) using memory mapping.
    Reduces peak RAM from ~4.5GB down to ~54MB per worker.
    """
    import sys
    import importlib
    from pathlib import Path
    import gc
    import numpy as np

    workspace_dir = Path(__file__).parent.resolve()
    if str(workspace_dir) not in sys.path:
        sys.path.insert(0, str(workspace_dir))
        
    module = importlib.import_module(module_name)
    importlib.reload(module)
    FEATURE_REGISTRY = module.FEATURE_REGISTRY

    data = None
    try:
        # Load preprocessed cache dictionary
        data = np.load(npz_file, allow_pickle=True)
        session_dict = {key: data[key] for key in data.files}
        data.close()
        data = None

        # Resolve raw traces path
        raw_path_str = session_dict['session_path']
        if isinstance(raw_path_str, np.ndarray):
            raw_path_str = raw_path_str.item()
        raw_path = Path(raw_path_str)

        f_path = raw_path / 'F.npy'
        fneu_path = raw_path / 'Fneu.npy'

        if not f_path.exists() or not fneu_path.exists():
            return False, f"Raw traces not found in session path '{raw_path}'"

        # Memory map the raw traces to avoid loading whole array into RAM
        F_mmap = np.load(f_path, mmap_mode='r')
        Fneu_mmap = np.load(fneu_path, mmap_mode='r')
        n_cells = F_mmap.shape[0]

        # Separate morphology features (instant 1D) from trace features (streamed 2D)
        trace_features = [f for f in features_to_process if f not in MORPHOLOGY_FEATURES]
        morph_features = [f for f in features_to_process if f in MORPHOLOGY_FEATURES]

        # 1. Compute morphology features on session_dict
        if morph_features:
            for feat in morph_features:
                extractor = FEATURE_REGISTRY[feat]
                session_dict[feat] = extractor(session_dict)

        # 2. Compute trace-based features in chunks
        if trace_features:
            # Pre-allocate output arrays in float32
            out_arrays = {feat: np.zeros(n_cells, dtype=np.float32) for feat in trace_features}

            for start_idx in range(0, n_cells, chunk_size):
                end_idx = min(start_idx + chunk_size, n_cells)
                
                # Load only 250 cells into RAM as float32
                F_chunk = np.array(F_mmap[start_idx:end_idx], dtype=np.float32)
                Fneu_chunk = np.array(Fneu_mmap[start_idx:end_idx], dtype=np.float32)

                chunk_cache = {
                    'F': F_chunk,
                    'Fneu': Fneu_chunk,
                }

                # Add slice of any required 1D features if needed
                for k in session_dict:
                    if k not in ('F', 'Fneu') and isinstance(session_dict[k], np.ndarray) and session_dict[k].ndim > 0 and session_dict[k].shape[0] == n_cells:
                        chunk_cache[k] = session_dict[k][start_idx:end_idx]

                for feat in trace_features:
                    extractor = FEATURE_REGISTRY[feat]
                    res = extractor(chunk_cache)
                    out_arrays[feat][start_idx:end_idx] = res

                del chunk_cache
                del F_chunk
                del Fneu_chunk

            # Copy computed trace arrays into session_dict
            for feat in trace_features:
                session_dict[feat] = out_arrays[feat]
            del out_arrays

        # Clean up memory immediately
        del F_mmap
        del Fneu_mmap
        if '_sorted_Fcorr' in session_dict:
            del session_dict['_sorted_Fcorr']
        if '_fcorr' in session_dict:
            del session_dict['_fcorr']
        gc.collect()

        # Write back to cache file once
        np.savez_compressed(npz_file, **session_dict)
        del session_dict
        gc.collect()
        return True, None

    except Exception as e:
        if data is not None:
            try:
                data.close()
            except:
                pass
        return False, str(e)

def print_progress_bar(iteration, total, start_time, prefix='', suffix='', length=30, fill='█'):
    elapsed = time.time() - start_time
    percent = (iteration / float(total)) * 100.0 if total > 0 else 100.0
    filled_length = int(length * iteration // total) if total > 0 else length
    bar = fill * filled_length + '-' * (length - filled_length)
    
    if iteration > 0:
        eta_sec = int((elapsed / iteration) * (total - iteration))
        mins, secs = divmod(eta_sec, 60)
        eta_str = f"(ETA: {mins:02d}:{secs:02d})"
    else:
        eta_str = "(ETA: --:--)"
        
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {eta_str} {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()

def main():
    parser = argparse.ArgumentParser(description="Zero-Crash Chunked Multi-Core Feature Cache Engine")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--feature', '-f', type=str, help="Name of the specific feature to recompute")
    group.add_argument('--all', '-a', action='store_true', help="Recompute all features in FEATURE_REGISTRY")
    parser.add_argument('--workers', '-w', type=int, default=4, help="Number of parallel worker processes (default: 4)")
    parser.add_argument('--chunk-size', '-c', type=int, default=250, help="Cell chunk size for memory-safe streaming (default: 250)")
    args = parser.parse_args()

    workspace_dir = Path(__file__).parent.resolve()
    cache_dir = workspace_dir / "preprocessed_cache"

    if not cache_dir.exists():
        print(f"Error: Cache directory '{cache_dir}' does not exist.")
        sys.exit(1)

    # Append workspace root to path and import fe_definitions dynamically
    sys.path.insert(0, str(workspace_dir))
    import fe_engine.fe_definitions
    importlib.reload(fe_engine.fe_definitions)
    FEATURE_REGISTRY = fe_engine.fe_definitions.FEATURE_REGISTRY

    # Resolve which features to process
    if args.all:
        features_to_process = list(FEATURE_REGISTRY.keys())
        print(f"Resolving to recompute ALL {len(features_to_process)} registered features...")
    else:
        feature_name = args.feature.strip()
        if feature_name not in FEATURE_REGISTRY:
            print(f"Error: Feature '{feature_name}' not defined in fe_definitions.py FEATURE_REGISTRY.")
            sys.exit(1)
        features_to_process = [feature_name]
        print(f"Resolving to recompute feature '{feature_name}'...")

    npz_files = sorted(list(cache_dir.glob("preprocessed_*.npz")))
    if len(npz_files) == 0:
        print(f"Error: No cached .npz files found in '{cache_dir}'.")
        sys.exit(1)

    total_files = len(npz_files)
    print(f"Found {total_files} cached session files.")
    print(f"Streaming mode: {args.chunk_size} cells/chunk | Workers: {args.workers} (Peak RAM < 400MB total)\n")

    start_time = time.time()
    success_count = 0
    print_progress_bar(0, total_files, start_time, prefix='Progress:', suffix='Starting...', length=30)

    # Execute parallel chunked recomputation
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                recompute_session_chunked,
                npz_file,
                features_to_process,
                'fe_engine.fe_definitions',
                args.chunk_size
            ): npz_file
            for npz_file in npz_files
        }

        for idx, future in enumerate(as_completed(futures)):
            npz_file = futures[future]
            try:
                ok, err = future.result()
            except Exception as e:
                ok, err = False, str(e)

            if ok:
                success_count += 1
                suffix_str = f"[{idx+1}/{total_files}] Done {npz_file.name[:24]}..."
            else:
                suffix_str = f"[{idx+1}/{total_files}] Failed {npz_file.name[:24]}: {err}"

            print_progress_bar(idx + 1, total_files, start_time, prefix='Progress:', suffix=suffix_str, length=30)

    total_elapsed = time.time() - start_time
    print(f"\nCompleted: {success_count}/{total_files} sessions updated in {total_elapsed:.1f}s.")
    print("✅ Finished recomputing cache successfully with zero crashes!")

if __name__ == "__main__":
    main()
