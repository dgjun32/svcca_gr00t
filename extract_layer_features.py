import os
import sys
# Disable flash attention to avoid ABI compatibility issues
# Add gr00t to path if needed
gr00t_path = "/home/dongjun/GR00T_Training"
if gr00t_path not in sys.path:
    sys.path.insert(0, gr00t_path)
    
    
import torch
from torch import nn
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

from feature_extraction_utils import extract_features_from_forward

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/dongjun/gr00t_analysis/svcca_datasets/trash_separation_3_objs_unseen_251124_depth_1")
    parser.add_argument("--model_path", type=str, default="nvidia/GR00T-N1.5-3B")
    parser.add_argument("--cache_path", type=str, default="/home/dongjun/gr00t_analysis/cache")
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
    
    # Create HDF5 file path
    os.makedirs(args.cache_path, exist_ok=True)
    dataset_name = os.path.basename(args.dataset_path.rstrip('/'))
    model_name = args.model_path.split("/")[-2]
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
    
    # Extract and save features
    with torch.no_grad():  # Disable gradient computation to save memory
        for idx, sample in enumerate(tqdm(train_dataset, desc="Extracting features")):
            sample = collate([sample], eagle_processor)
            features_dict = extract_features_from_forward(model, sample)
            
            # Incrementally write to HDF5 file (compressed)
            group = h5_file.create_group(f"sample_{idx}")
            for key, value in features_dict.items():
                if isinstance(value, torch.Tensor):
                    # Convert bfloat16 to float32 before numpy conversion
                    if value.dtype == torch.bfloat16:
                        value = value.float()
                    # Save as compressed dataset
                    group.create_dataset(
                        key, 
                        data=value.cpu().numpy(), 
                        compression="gzip", 
                        compression_opts=4  # compression level 1-9
                    )
                else:
                    # Store non-tensor values as attributes
                    group.attrs[key] = value
            
            # Memory cleanup every iteration
            del features_dict, sample
            
            # Periodic GPU cache clearing and garbage collection
            if (idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                h5_file.flush()  # Flush HDF5 buffer to disk
                
                # Log memory status every 50 samples
                if (idx + 1) % 50 == 0 and torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    tqdm.write(f"[Sample {idx+1}] GPU memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
    
    # Close HDF5 file
    h5_file.close()
    print(f"\nFeatures saved to {h5_path}")
    
    # Final memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleanup completed.")
        
        

