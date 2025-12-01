import os
import sys
# Disable flash attention to avoid ABI compatibility issues
# Add gr00t to path if needed
gr00t_path = "/home/dongjun/GR00T_Training"
if gr00t_path not in sys.path:
    sys.path.insert(0, gr00t_path)
    
    
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers.feature_extraction_utils import BatchFeature
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import build_eagle_processor
from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.schema import EmbodimentTag
from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.model.transforms import DefaultDataCollator
from gr00t.model.transforms import collate
import wandb
import numpy as np
import h5py
from tqdm import tqdm
import gc
from functools import partial

from feature_extraction_utils import extract_features_from_forward, IntermediateFeatureExtractor

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/dongjun/gr00t_analysis/svcca_datasets/trash_separation_3_objs_unseen_251124_depth_1")
    parser.add_argument("--model_path", type=str, default="nvidia/GR00T-N1.5-3B")
    parser.add_argument("--cache_path", type=str, default="/home/dongjun/gr00t_analysis/cache")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for feature extraction")
    args = parser.parse_args()

    # load model and processor
    model = GR00T_N1_5.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            tune_llm=True,  # backbone's LLM
            tune_visual=True,  # backbone's vision tower
            tune_projector=False,  # action head's projector
            tune_diffusion_model=False,  # action head's DiT
            torch_dtype=torch.bfloat16,
        )
    model = model.to("cuda")
    model.eval()  # Set to evaluation mode (disable dropout, etc.)
    
    # Create feature extractor once and register hooks
    print("Initializing feature extractor...")
    feature_extractor = IntermediateFeatureExtractor(model) 
    feature_extractor.register_hooks()
    print(f"Registered hooks on model layers")
        
    # Get EAGLE VLM backbone
    #eagle_model = model.backbone.eagle_model

    # Get eagle processor
    eagle_processor = build_eagle_processor(
        '/home/dongjun/GR00T_Training/gr00t/model/backbone/eagle2_hg_model',
        )
    eagle_processor.tokenizer.padding_side = "left"

    # get dataset
    data_config_cls = DATA_CONFIG_MAP["bimanual_piper"]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()
    embodiment_tag = EmbodimentTag("new_embodiment")
    
    train_dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
        video_backend="torchvision_av",
    )
    
    # Create DataLoader for batch processing
    def collate_fn(batch):
        return collate(batch, eagle_processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Keep order for reproducibility
        num_workers=0,  # Use 0 to avoid multiprocessing issues
        collate_fn=collate_fn,
    )
    
    total_samples = len(train_dataset)
    print(f"Total samples: {total_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {len(train_loader)}")
    
    # Create HDF5 file path
    os.makedirs(args.cache_path, exist_ok=True)
    dataset_name = os.path.basename(args.dataset_path.rstrip('/'))
    model_name = args.model_path.split("/")[-2] + "-" + args.model_path.split("/")[-1]
    h5_filename = f"model_{model_name}_dataset_{dataset_name}.hdf5"
    h5_path = os.path.join(args.cache_path, h5_filename)
    
    # Remove existing file if it exists
    if os.path.exists(h5_path):
        print(f"Warning: {h5_path} already exists. Removing it.")
        os.remove(h5_path)
    
    # Create HDF5 file for incremental writing
    print(f"Creating HDF5 file: {h5_path}")
    h5_file = h5py.File(h5_path, "w")
    
    # Print initial GPU memory status
    if torch.cuda.is_available():
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Extract and save features in batch mode
    h5_datasets = {}  # Will store HDF5 dataset objects for each layer
    current_idx = 0  # Track total samples processed
    
    try:
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Extracting features")):
        # Extract features for the batch (reuse extractor for efficiency)
            with torch.no_grad():  # Disable gradient computation to save memory
                features_dict = extract_features_from_forward(model, batch, extractor=feature_extractor)

            # Get actual batch size (important for last batch which might be smaller)
            current_batch_size = args.batch_size
            
            # First batch: create resizable datasets for each layer
            if batch_idx == 0:
                print(f"\nCreating HDF5 datasets for {len(features_dict)} layers...")
                for layer_name, feature_tensor in features_dict.items():
                    if isinstance(feature_tensor, torch.Tensor):
                        # Convert to float16 for smaller file size (half of float32)
                        if feature_tensor.dtype == torch.bfloat16:
                            feature_tensor = feature_tensor.to(torch.float16)
                        elif feature_tensor.dtype == torch.float32:
                            feature_tensor = feature_tensor.to(torch.float16)
                        
                        feature_array = feature_tensor.cpu().numpy()
                        
                        # Handle vision encoder features: (batch_size * num_images, L, D) -> (batch_size, num_images, L, D)
                        # Vision encoder processes images separately, so first dim is batch_size * num_images
                        if feature_array.shape[0] != current_batch_size and feature_array.shape[0] % current_batch_size == 0:
                            num_images = feature_array.shape[0] // current_batch_size
                            # Reshape: (batch_size * num_images, L, D) -> (batch_size, num_images * L, D)
                            # This keeps the batch dimension first
                            original_shape = feature_array.shape
                            feature_array = feature_array.reshape(current_batch_size, num_images , original_shape[1], original_shape[2])
                            tqdm.write(f"  Reshaped vision feature '{layer_name}': {original_shape} -> {feature_array.shape}")
                        
                        feature_shape = feature_array.shape  # (batch_size, L, D, ...)
                        
                        # Create resizable dataset with maxshape
                        # shape[0] is resizable (for total samples), rest are fixed
                        maxshape = (total_samples,) + feature_shape[1:]
                        
                        h5_datasets[layer_name] = h5_file.create_dataset(
                            layer_name,
                            shape=feature_shape,  # Initial shape with first batch
                            maxshape=maxshape,  # Maximum shape (total_samples, L, D, ...)
                            dtype=np.float16,  # Explicitly use float16
                            compression="gzip",
                            compression_opts=4,
                            chunks=True  # Enable chunking for resizable datasets
                        )
                        
                        # Write first batch
                        h5_datasets[layer_name][0:current_batch_size] = feature_array
                        tqdm.write(f"  Created dataset '{layer_name}': shape {maxshape}, dtype float16")
                
                current_idx = current_batch_size
            
            # Subsequent batches: resize and append
            else:
                for layer_name, feature_tensor in features_dict.items():
                    if isinstance(feature_tensor, torch.Tensor):
                        # Convert to float16 for smaller file size
                        if feature_tensor.dtype == torch.bfloat16:
                            feature_tensor = feature_tensor.to(torch.float16)
                        elif feature_tensor.dtype == torch.float32:
                            feature_tensor = feature_tensor.to(torch.float16)
                        
                        feature_array = feature_tensor.cpu().numpy()
                        
                        # Handle vision encoder features: (batch_size * num_images, L, D) -> (batch_size, num_images * L, D)
                        if feature_array.shape[0] != current_batch_size and feature_array.shape[0] % current_batch_size == 0:
                            num_images = feature_array.shape[0] // current_batch_size
                            original_shape = feature_array.shape
                            feature_array = feature_array.reshape(current_batch_size, num_images , original_shape[1], original_shape[2])
                        
                        # Resize dataset to accommodate new batch
                        new_size = current_idx + current_batch_size
                        h5_datasets[layer_name].resize(new_size, axis=0)
                        
                        # Write new batch
                        h5_datasets[layer_name][current_idx:new_size] = feature_array
                
                current_idx += current_batch_size
            
            # Memory cleanup every batch
            del features_dict, batch
            
            # Periodic GPU cache clearing and garbage collection
            if (batch_idx + 1) % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                h5_file.flush()  # Flush HDF5 buffer to disk
                
                # Log memory status
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    tqdm.write(f"[Batch {batch_idx+1}/{len(train_loader)}] "
                                f"Samples: {current_idx}/{total_samples} | "
                                f"GPU memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
            
            # Normal completion
            print(f"\nAll batches processed successfully!")
            print(f"Total samples: {current_idx}/{total_samples}")
        
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("⚠️  Interrupted by user (Ctrl+C)!")
        print(f"Progress saved: {current_idx}/{total_samples} samples processed ({current_idx/total_samples*100:.1f}%)")
        print("="*60)
        
    finally:
        # Always execute cleanup, whether completed or interrupted
        print("\nCleaning up...")
        
        # Clean up feature extractor hooks
        print("  - Removing feature extractor hooks...")
        feature_extractor.clear_hooks()
        
        # Flush and close HDF5 file
        print("  - Flushing and closing HDF5 file...")
        h5_file.flush()
        h5_file.close()
        print(f"  - Features saved to: {h5_path}")
        
        # Final memory cleanup
        print("  - Clearing GPU memory...")
        torch.cuda.empty_cache()
        gc.collect()
        
        print("\n✓ Cleanup completed. File is safely saved with all processed data.")
        
        

