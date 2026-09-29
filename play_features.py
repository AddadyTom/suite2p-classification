import os
import sys
import argparse
import json
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import importlib

# Ensure workspace root is in python path
workspace_root = Path(__file__).parent.resolve()
sys.path.append(str(workspace_root))

from fe_engine.fe_definitions import ACTIVE_FEATURES, FEATURE_REGISTRY, safe_eval_formula
from fe_engine.fe_loop_runner import load_preprocessed_data, extract_features_dataset, run_cross_validation, analyze_errors_and_shap


def evaluate_formula_on_sessions(sessions, name, expression):
    """
    Evaluate a custom mathematical expression on the precomputed feature arrays
    for each session, saving the result under session[name].
    """
    for session in sessions:
        # Build namespace using 1D numpy arrays in the session dictionary
        ns = {}
        for key in session.keys():
            val = session[key]
            if isinstance(val, np.ndarray) and len(val.shape) == 1:
                ns[key] = val
        
        try:
            # Evaluate expression securely using AST parser
            result = safe_eval_formula(expression, ns)
            session[name] = result
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression}' on session {session['session_name']}: {e}")


def recompute_session_file(f, feature_name, extractor_func):
    """
    Worker function to load raw traces, recompute a single feature, and update the npz file on disk.
    """
    try:
        # Load existing npz data
        data = np.load(f, allow_pickle=True)
        session_dict = {key: data[key] for key in data.files}
        
        # Resolve raw path
        raw_path_str = session_dict['session_path']
        if isinstance(raw_path_str, np.ndarray):
            raw_path_str = raw_path_str.item()
        raw_path = Path(raw_path_str)
        
        if not raw_path.exists():
            return False, f"Raw session folder '{raw_path}' does not exist."
            
        # Load raw traces using memmap to save memory
        F_arr = np.load(raw_path / 'F.npy', mmap_mode='r')
        Fneu_arr = np.load(raw_path / 'Fneu.npy', mmap_mode='r')
        
        session_dict['F'] = F_arr
        session_dict['Fneu'] = Fneu_arr
        
        # Re-extract the feature values
        new_val = extractor_func(session_dict)
        
        # Remove raw traces from dictionary
        del session_dict['F']
        del session_dict['Fneu']
        
        # Update cache dictionary
        session_dict[feature_name] = new_val
        
        # Save back to same file (compressed)
        np.savez_compressed(f, **session_dict)
        return True, raw_path.name
    except Exception as e:
        return False, f"{f.name}: {e}"


def recompute_feature_on_disk(feature_name, cache_dir):
    """
    Recompute a specific feature from raw traces in parallel and update the preprocessed cache.
    """
    global FEATURE_REGISTRY
    
    # Reload definitions in case they modified the extraction code
    reload_definitions()
    
    cache_path = Path(cache_dir)
    npz_files = list(cache_path.glob("preprocessed_*.npz"))
    
    if len(npz_files) == 0:
        print("No cache files found to recompute.")
        return False
        
    if feature_name not in FEATURE_REGISTRY:
        print(f"Error: Feature '{feature_name}' not defined in fe_definitions.py FEATURE_REGISTRY.")
        return False
        
    extractor = FEATURE_REGISTRY[feature_name]
    print(f"Recomputing feature '{feature_name}' in parallel for {len(npz_files)} sessions...")
    
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=-1)(
        delayed(recompute_session_file)(f, feature_name, extractor)
        for f in npz_files
    )
    
    success_count = sum(1 for r in results if r[0])
    print(f"Recomputation finished. Successfully updated cache for {success_count}/{len(npz_files)} sessions.")
    
    errors = [r[1] for r in results if not r[0]]
    if errors:
        print("Errors encountered:")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors.")
            
    return success_count > 0


def reload_definitions():
    """
    Reload fe_definitions module dynamically to pick up any python code updates.
    """
    global FEATURE_REGISTRY
    try:
        import fe_engine.fe_definitions
        importlib.reload(fe_engine.fe_definitions)
        from fe_engine.fe_definitions import FEATURE_REGISTRY as new_registry
        FEATURE_REGISTRY.clear()
        FEATURE_REGISTRY.update(new_registry)
        print("Successfully reloaded fe_engine/fe_definitions.py.")
    except Exception as e:
        print(f"Error reloading fe_definitions.py: {e}")


def print_comparison(current, baseline):
    """
    Prints a formatted comparison table between the current run and the baseline.
    """
    if baseline is None:
        print("\n=== Current Performance ===")
        print(f"  F1 Score:  {current['f1']:.4f} ± {current['f1_std']:.4f}")
        print(f"  Precision: {current['precision']:.4f} ± {current['precision_std']:.4f}")
        print(f"  Recall:    {current['recall']:.4f} ± {current['recall_std']:.4f}")
        return

    diff_f1 = current['f1'] - baseline['f1']
    diff_p = current['precision'] - baseline['precision']
    diff_r = current['recall'] - baseline['recall']
    
    print("\n" + "="*60)
    print(" PERFORMANCE COMPARISON AGAINST BASELINE")
    print("="*60)
    print(f" Metric    | Baseline | Current  | Difference")
    print(f" ----------|----------|----------|-----------")
    print(f" F1-Score  |  {baseline['f1']:.4f}  |  {current['f1']:.4f}  | {diff_f1:+.4f}")
    print(f" Precision |  {baseline['precision']:.4f}  |  {current['precision']:.4f}  | {diff_p:+.4f}")
    print(f" Recall    |  {baseline['recall']:.4f}  |  {current['recall']:.4f}  | {diff_r:+.4f}")
    print("="*60)


def start_interactive_shell(sessions, active_features, cache_dir):
    """
    Interactive command-line shell for feature play.
    """
    baseline_metrics = None
    baseline_features = None
    
    # Store custom formulas in memory
    custom_formulas = {}
    
    print("\n" + "="*70)
    print(" Welcome to the Suite2p Feature Playground!")
    print(" Type 'help' to see the available commands.")
    print("="*70)
    
    while True:
        try:
            cmd_line = input("\nplay-features> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting playground.")
            break
            
        if not cmd_line:
            continue
            
        tokens = cmd_line.split()
        cmd = tokens[0].lower()
        args = tokens[1:]
        
        if cmd in ("exit", "quit"):
            print("Goodbye!")
            break
            
        elif cmd == "help":
            print("Available commands:")
            print("  list                       List all registry features, custom features, and their active status")
            print("  active                     Show currently active features")
            print("  add <name> = <expr>        Add a custom feature via math expression (e.g. add f_ratio = solidity * mrs)")
            print("  drop <feature>             Exclude a feature from active evaluation")
            print("  use <feature>              Include a feature in active evaluation")
            print("  run                        Run 5-fold cross-validation and output metrics & importances")
            print("  baseline                   Save current active features and metrics as baseline for comparison")
            print("  recompute <feature>        Reload code and recompute a feature from raw GCaMP traces (updates disk cache)")
            print("  reload                     Reload fe_definitions.py dynamically")
            print("  save                       Write current active features list to fe_definitions.py")
            print("  exit / quit                Exit the playground")
            
        elif cmd == "list":
            print("\nAvailable Features:")
            print(f" {'Feature Name':<28} | {'Category/Formula':<35} | {'Status':<10}")
            print("-" * 80)
            
            # Print precomputed/registry features
            for name in sorted(FEATURE_REGISTRY.keys()):
                status = "ACTIVE" if name in active_features else "inactive"
                print(f"  {name:<28} | {'Registry Extractor':<35} | {status:<10}")
                
            # Print custom formulas
            for name, expr in custom_formulas.items():
                status = "ACTIVE" if name in active_features else "inactive"
                print(f"  {name:<28} | {expr:<35} | {status:<10}")
                
        elif cmd == "active":
            print(f"\nActive features ({len(active_features)}):")
            for idx, f in enumerate(active_features):
                print(f"  {idx+1:2d}. {f}")
                
        elif cmd == "add":
            formula_str = " ".join(args)
            if "=" not in formula_str:
                print("Error: Formula must be of the form 'name = expression'. Example: add f_ratio = solidity * mrs")
                continue
            name, expr = formula_str.split("=", 1)
            name = name.strip()
            expr = expr.strip()
            
            if not name or not expr:
                print("Error: Invalid name or expression.")
                continue
                
            print(f"Evaluating formula '{expr}' for custom feature '{name}'...")
            try:
                evaluate_formula_on_sessions(sessions, name, expr)
                custom_formulas[name] = expr
                if name not in active_features:
                    active_features.append(name)
                print(f"✅ Success: Feature '{name}' is now active!")
            except Exception as e:
                print(f"❌ Error: {e}")
                
        elif cmd == "drop":
            if not args:
                print("Error: Please specify a feature name to drop.")
                continue
            target = args[0]
            if target in active_features:
                active_features.remove(target)
                print(f"Dropped feature '{target}' from active list.")
            else:
                print(f"Feature '{target}' is not in the active list.")
                
        elif cmd == "use":
            if not args:
                print("Error: Please specify a feature name to include.")
                continue
            target = args[0]
            if target in active_features:
                print(f"Feature '{target}' is already active.")
            elif target in FEATURE_REGISTRY or target in custom_formulas:
                active_features.append(target)
                print(f"Added feature '{target}' to active list.")
            else:
                print(f"Feature '{target}' not found in registry or custom formulas. Use 'add' to create it.")
                
        elif cmd == "run":
            if not active_features:
                print("Error: No active features selected. Use 'use <feature>' or 'add' to activate some.")
                continue
                
            print(f"Compiling dataset for CV using {len(active_features)} active features...")
            try:
                X, y, groups = extract_features_dataset(sessions, active_features)
                print(f"Dataset compiled: X={X.shape}, y={y.shape}. Running GroupKFold...")
                cv_summary, y_true, y_pred, y_prob, shap_values = run_cross_validation(X, y, groups, active_features)
                
                print_comparison(cv_summary, baseline_metrics)
                
                # Show SHAP importance
                fp_culprits, fn_culprits, importance = analyze_errors_and_shap(X, y_true, y_pred, shap_values, active_features)
                print("\n=== Top Feature Importances (SHAP) ===")
                for rank, (name, val) in enumerate(importance[:10]):
                    print(f"  {rank+1:2d}. {name:<25} : {val:.5f}")
                    
                if fp_culprits:
                    print("\n=== Top False Positive Contributors (Noise predicted as Cells) ===")
                    print(f"  {'Feature':<22} | {'SHAP Impact':<12} | {'FP Mean':<10} | {'TN Mean':<10}")
                    for c in fp_culprits:
                        print(f"  {c['feature']:<22} | {c['shap_diff']:+12.4f} | {c['fp_mean_val']:10.4f} | {c['tn_mean_val']:10.4f}")
                        
                if fn_culprits:
                    print("\n=== Top False Negative Contributors (Cells predicted as Noise) ===")
                    print(f"  {'Feature':<22} | {'SHAP Impact':<12} | {'FN Mean':<10} | {'TP Mean':<10}")
                    for c in fn_culprits:
                        print(f"  {c['feature']:<22} | {c['shap_diff']:+12.4f} | {c['fn_mean_val']:10.4f} | {c['tp_mean_val']:10.4f}")
                        
            except Exception as e:
                print(f"❌ Error during evaluation: {e}")
                
        elif cmd == "baseline":
            if not active_features:
                print("Error: No active features to run for baseline.")
                continue
            print("Running CV to establish new baseline...")
            try:
                X, y, groups = extract_features_dataset(sessions, active_features)
                cv_summary, _, _, _, _ = run_cross_validation(X, y, groups, active_features)
                baseline_metrics = cv_summary
                baseline_features = active_features.copy()
                print(f"✅ Baseline saved! F1-Score: {baseline_metrics['f1']:.4f}")
            except Exception as e:
                print(f"❌ Error saving baseline: {e}")
                
        elif cmd == "recompute":
            if not args:
                print("Error: Please specify a feature name to recompute.")
                continue
            target = args[0]
            success = recompute_feature_on_disk(target, cache_dir)
            if success:
                # Reload the sessions cache to read the new npz files
                print("Reloading preprocessed cache from disk...")
                sessions = load_preprocessed_data(cache_dir)
                print("Cache reloaded successfully!")
                
        elif cmd == "reload":
            reload_definitions()
            
        elif cmd == "save":
            # Save the current active features to fe_definitions.py
            if not active_features:
                print("Error: No active features to save.")
                continue
                
            # Filter out custom features that aren't defined in fe_definitions.py registry
            reg_active = [f for f in active_features if f in FEATURE_REGISTRY]
            if len(reg_active) != len(active_features):
                print(f"Warning: Custom features {set(active_features) - set(reg_active)} cannot be saved to fe_definitions.py because they are math formulas.")
                
            confirm = input(f"Are you sure you want to write these {len(reg_active)} features as ACTIVE_FEATURES in fe_definitions.py? (y/n): ").strip().lower()
            if confirm == 'y':
                try:
                    from fe_engine.fe_loop_runner import update_definitions_file
                    update_definitions_file(reg_active)
                    print("✅ Active features list updated in fe_definitions.py.")
                except Exception as e:
                    print(f"Error updating file: {e}")
            else:
                print("Save cancelled.")
                
        else:
            print(f"Unknown command: '{cmd}'. Type 'help' to see list of commands.")


def main():
    parser = argparse.ArgumentParser(description="Interactive Feature Curation & Play Yard")
    parser.add_argument('--data', type=str, default=".", help="Workspace path containing datasets")
    parser.add_argument('--source', type=str, default="/mnt/other_ubunthu/mnt/data", help="Raw dataset root directory")
    parser.add_argument('--preset', type=str, default="definitions", choices=["definitions", "regular", "rich"], help="Starting feature set")
    parser.add_argument('--exclude', nargs='+', default=[], help="List of features to exclude from the preset")
    parser.add_argument('--include', nargs='+', default=[], help="List of features to include in the preset")
    parser.add_argument('--formula', action='append', default=[], help="Add math formulas, e.g. --formula 'log_npix = log(npix + 1)'")
    parser.add_argument('--recompute', type=str, default=None, help="Recompute specific feature from raw traces in parallel, then exit")
    parser.add_argument('--interactive', action='store_true', help="Open the interactive feature playground CLI shell")
    args = parser.parse_args()

    cache_dir = Path(args.data) / "preprocessed_cache"
    
    # Check if we just want to run recomputation and exit
    if args.recompute:
        success = recompute_feature_on_disk(args.recompute, cache_dir)
        sys.exit(0 if success else 1)
        
    print("Loading preprocessed cache...")
    try:
        sessions = load_preprocessed_data(cache_dir)
    except Exception as e:
        print(f"Error loading cache: {e}")
        print("Please ensure preprocessed_cache folder exists and contains valid .npz files.")
        sys.exit(1)
        
    # Resolve starting active features
    if args.preset == "definitions":
        active_features = ACTIVE_FEATURES.copy()
    elif args.preset == "regular":
        from apply_AI import FEATURE_NAMES_25
        active_features = FEATURE_NAMES_25.copy()

    elif args.preset == "rich":
        from apply_AI import FEATURE_NAMES_38
        active_features = FEATURE_NAMES_38.copy()
        
    # Apply CLI exclusions/inclusions
    for exc in args.exclude:
        if exc in active_features:
            active_features.remove(exc)
            print(f"Excluded feature '{exc}' (CLI)")
            
    for inc in args.include:
        if inc not in active_features:
            active_features.append(inc)
            print(f"Included feature '{inc}' (CLI)")
            
    # Apply CLI math formulas
    for formula in args.formula:
        if "=" not in formula:
            print(f"Warning: Ignoring invalid formula '{formula}'. Must be 'name = expression'.")
            continue
        name, expr = formula.split("=", 1)
        name = name.strip()
        expr = expr.strip()
        try:
            evaluate_formula_on_sessions(sessions, name, expr)
            if name not in active_features:
                active_features.append(name)
            print(f"Added feature '{name}' via formula (CLI)")
        except Exception as e:
            print(f"Error evaluating formula '{formula}': {e}")
            
    # Open interactive mode or run a single execution
    if args.interactive:
        start_interactive_shell(sessions, active_features, cache_dir)
    else:
        # Run standard 5-fold CV
        print(f"\nRunning 5-fold CV with {len(active_features)} active features:")
        X, y, groups = extract_features_dataset(sessions, active_features)
        cv_summary, y_true, y_pred, y_prob, shap_values = run_cross_validation(X, y, groups, active_features)
        
        print("\n=== Cross-Validation Results ===")
        print(f"  F1 Score:  {cv_summary['f1']:.4f} ± {cv_summary['f1_std']:.4f}")
        print(f"  Precision: {cv_summary['precision']:.4f} ± {cv_summary['precision_std']:.4f}")
        print(f"  Recall:    {cv_summary['recall']:.4f} ± {cv_summary['recall_std']:.4f}")
        
        # Save a report
        report_path = Path(args.data) / "fe_report.md"
        fp_culprits, fn_culprits, importance = analyze_errors_and_shap(X, y_true, y_pred, shap_values, active_features)
        
        # Check against baseline
        baseline_json_path = Path(args.data) / "fe_baseline.json"
        if baseline_json_path.exists():
            with open(baseline_json_path, 'r') as f:
                baseline = json.load(f)
            diff_f1 = cv_summary['f1'] - baseline['f1']
            print(f"\nComparison with baseline (F1={baseline['f1']:.4f}): Diff = {diff_f1:+.4f}")


if __name__ == "__main__":
    main()
