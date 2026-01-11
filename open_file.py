import h5py
import numpy as np
import torch
from typing import List, Union


def get_layers(hdf5_path: str) -> List[str]:
    """
    Get all layer names (keys) from an HDF5 file.
    
    Args:
        hdf5_path: Path to the HDF5 file
        
    Returns:
        List of layer names
    """
    with h5py.File(hdf5_path, 'r') as f:
        layers = list(f.keys())
    return layers


def load_feature(hdf5_path: str, layer_name: str) -> np.ndarray:
    """
    Load features from a specific layer in an HDF5 file.
    
    Args:
        hdf5_path: Path to the HDF5 file
        layer_name: Name of the layer to load
        
    Returns:
        Features as a numpy array
    """
    with h5py.File(hdf5_path, 'r') as f:
        if layer_name not in f:
            available_layers = list(f.keys())
            raise KeyError(f"Layer '{layer_name}' not found in HDF5 file. Available layers: {available_layers}")
        
        feature = f[layer_name][:]
    
    return feature


def compare_features(hdf5_path1: str, hdf5_path2: str, device: str = None) -> None:
    """
    Compare features from two HDF5 files layer-by-layer and print the norm of differences.
    
    Args:
        hdf5_path1: Path to the first model's HDF5 file
        hdf5_path2: Path to the second model's HDF5 file
        device: Device to use ('cuda', 'cpu', etc.). Auto-selected if None
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Model 1: {hdf5_path1}")
    print(f"Model 2: {hdf5_path2}")
    print("=" * 80)
    
    # Get layer lists from both files
    layers1 = get_layers(hdf5_path1)
    layers2 = get_layers(hdf5_path2)
    
    # Find common layers
    common_layers = [
        'backbone.llm.layer_0_step_0',
        'backbone.llm.layer_1_step_0',
        'backbone.llm.layer_2_step_0',
        'backbone.llm.layer_3_step_0',
        'backbone.llm.layer_4_step_0',
        'backbone.llm.layer_5_step_0',
        'backbone.llm.layer_6_step_0',
        'backbone.llm.layer_7_step_0',
        'backbone.llm.layer_8_step_0',
        'backbone.llm.layer_9_step_0',
        'backbone.llm.layer_10_step_0',
        'backbone.llm.layer_11_step_0',
        'action_head.dit.layer_0_step_0',
        'action_head.dit.layer_1_step_0',
        'action_head.dit.layer_2_step_0',
        'action_head.dit.layer_3_step_0',
        'action_head.dit.layer_4_step_0',
        'action_head.dit.layer_5_step_0',
        'action_head.dit.layer_6_step_0',
        'action_head.dit.layer_7_step_0',
        'action_head.dit.layer_8_step_0',
        'action_head.dit.layer_9_step_0',
        'action_head.dit.layer_10_step_0',
        'action_head.dit.layer_11_step_0',
        'action_head.dit.layer_12_step_0',
        'action_head.dit.layer_13_step_0',
        'action_head.dit.layer_14_step_0',
        'action_head.dit.layer_15_step_0',
        'action_head.dit.layer_0_step_1',
        'action_head.dit.layer_1_step_1',
        'action_head.dit.layer_2_step_1',
        'action_head.dit.layer_3_step_1',
        'action_head.dit.layer_4_step_1',
        'action_head.dit.layer_5_step_1',
        'action_head.dit.layer_6_step_1',
        'action_head.dit.layer_7_step_1',
        'action_head.dit.layer_8_step_1',
        'action_head.dit.layer_9_step_1',
        'action_head.dit.layer_10_step_1',
        'action_head.dit.layer_11_step_1',
        'action_head.dit.layer_12_step_1',
        'action_head.dit.layer_13_step_1',
        'action_head.dit.layer_14_step_1',
        'action_head.dit.layer_15_step_1',
        'action_head.dit.layer_0_step_2',
        'action_head.dit.layer_1_step_2',
        'action_head.dit.layer_2_step_2',
        'action_head.dit.layer_3_step_2',
        'action_head.dit.layer_4_step_2',
        'action_head.dit.layer_5_step_2',
        'action_head.dit.layer_6_step_2',
        'action_head.dit.layer_7_step_2',
        'action_head.dit.layer_8_step_2',
        'action_head.dit.layer_9_step_2',
        'action_head.dit.layer_10_step_2',
        'action_head.dit.layer_11_step_2',
        'action_head.dit.layer_12_step_2',
        'action_head.dit.layer_13_step_2',
        'action_head.dit.layer_14_step_2',
        'action_head.dit.layer_15_step_2',
        'action_head.dit.layer_0_step_3',
        'action_head.dit.layer_1_step_3',
        'action_head.dit.layer_2_step_3',
        'action_head.dit.layer_3_step_3',
        'action_head.dit.layer_4_step_3',
        'action_head.dit.layer_5_step_3',
        'action_head.dit.layer_6_step_3',
        'action_head.dit.layer_7_step_3',
        'action_head.dit.layer_8_step_3',
        'action_head.dit.layer_9_step_3',
        'action_head.dit.layer_10_step_3',
        'action_head.dit.layer_11_step_3',
        'action_head.dit.layer_12_step_3',
        'action_head.dit.layer_13_step_3',
        'action_head.dit.layer_14_step_3',
        'action_head.dit.layer_15_step_3',
    ]
    
    print(f"Comparing {len(common_layers)} common layers...\n")
    
    # Compare each layer
    results = []
    print_dict = {}
    for i, layer_name in enumerate(common_layers, 1):
        try:
            # Load as NumPy arrays
            feature1_np = load_feature(hdf5_path1, layer_name)
            feature2_np = load_feature(hdf5_path2, layer_name)
            
            # Check shapes
            if feature1_np.shape != feature2_np.shape:
                print(f"[{i}/{len(common_layers)}] {layer_name}")
                print(f"  [WARNING] Shape mismatch! Model1={feature1_np.shape}, Model2={feature2_np.shape}\n")
                continue
            
            # Convert to PyTorch tensors and move to device
            feature1 = torch.from_numpy(feature1_np).to(device)
            feature2 = torch.from_numpy(feature2_np).to(device)
            
            # Normalize along axis=-1 (make each vector a unit vector)
            feature1_norm_per_vec = torch.norm(feature1, dim=-1, keepdim=True)
            feature2_norm_per_vec = torch.norm(feature2, dim=-1, keepdim=True)
            
            feature1_normalized = feature1 / (feature1_norm_per_vec + 1e-8)
            feature2_normalized = feature2 / (feature2_norm_per_vec + 1e-8)
            
            # Calculate difference between normalized features
            diff = feature1_normalized - feature2_normalized
            
            # Calculate L2 norm of the difference
            diff_norm = torch.norm(diff).item()
            
            # Mean absolute difference (per element)
            mean_abs_diff = torch.mean(torch.abs(diff)).item()
            
            # Original feature norms (for reference)
            feature1_norm = torch.mean(feature1_norm_per_vec).item()
            feature2_norm = torch.mean(feature2_norm_per_vec).item()
            
            # Store result (move to CPU)
            print_dict[layer_name] = mean_abs_diff
            
            # Store result
            result = {
                'layer': layer_name,
                'shape': feature1_np.shape,
                'diff_norm': diff_norm,
                'mean_abs_diff': mean_abs_diff,
                'feature1_norm': feature1_norm,
                'feature2_norm': feature2_norm
            }
            results.append(result)
            
            # Print per-layer results
            print(f"[{i}/{len(common_layers)}] {layer_name}")
            print(f"  Shape: {feature1_np.shape}")
            print(f"  Diff Norm (normalized features): {diff_norm:.6f}")
            print(f"  Mean |Diff|: {mean_abs_diff:.6f}")
            print(f"  Avg Feature Norms: Model1={feature1_norm:.6f}, Model2={feature2_norm:.6f}")
            print()
            
            # Free GPU memory
            del feature1, feature2, feature1_normalized, feature2_normalized, diff
            torch.cuda.empty_cache() if device == 'cuda' else None
            
        except Exception as e:
            print(f"[{i}/{len(common_layers)}] {layer_name}")
            print(f"  [ERROR] {e}\n")
    
    print(print_dict)
    # Print summary statistics
    if results:
        print("=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        diff_norms = [r['diff_norm'] for r in results]
        
        print(f"Difference Norm (between normalized features):")
        print(f"  Min:    {min(diff_norms):.6f}")
        print(f"  Max:    {max(diff_norms):.6f}")
        print(f"  Mean:   {np.mean(diff_norms):.6f}")
        print(f"  Median: {np.median(diff_norms):.6f}")
        print(f"  Std:    {np.std(diff_norms):.6f}")
        
        # Print layers with largest differences
        print()
        print("Top 5 layers with largest difference:")
        sorted_results = sorted(results, key=lambda x: x['diff_norm'], reverse=True)
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {result['layer']}: {result['diff_norm']:.6f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python open_file.py <model1_features.hdf5> <model2_features.hdf5>")
        print("\nExample:")
        print("  python open_file.py base_features.hdf5 finetuned_features.hdf5")
        sys.exit(1)
    
    hdf5_path1 = sys.argv[1]
    hdf5_path2 = sys.argv[2]
    
    compare_features(hdf5_path1, hdf5_path2)