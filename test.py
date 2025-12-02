import os
import sys
# Disable flash attention to avoid ABI compatibility issues
# Add gr00t to path if needed
gr00t_path = "/home/dongjun/GR00T_Training"
if gr00t_path not in sys.path:
    sys.path.insert(0, gr00t_path)

import torch
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


# LLM attention weights extraction only
from typing import Dict, Optional
from collections import OrderedDict


class LLMAttentionExtractor:
    """
    Extract attention weights from LLM layers in GR00T model.
    """
    
    def __init__(self, model):
        """
        Args:
            model: GR00T_N1_5 model instance
        """
        self.model = model
        self.attentions = OrderedDict()
        self.hooks = []
        self.step_counters = {}
        
    def _create_attention_hook(self, name: str):
        """Create a forward hook for self_attn to extract attention weights."""
        def hook(module, input, output):
            if name not in self.step_counters:
                self.step_counters[name] = 0
            
            step_name = f"{name}_step_{self.step_counters[name]}"
            
            # output = (hidden_states, attention_weights, ...)
            print(output[1])
            self.attentions[step_name] = output[1].detach()
            #if isinstance(output, tuple) and len(output) >= 2:
            #    attn_weights = output[1]
            #     if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
            #        self.attentions[step_name] = attn_weights.detach()
            
            self.step_counters[name] += 1
        return hook
    
    def register_hooks(self):
        """Register forward hooks on LLM self_attn layers."""
        self.clear_hooks()
        self.attentions.clear()
        
        # Enable output_attentions
        self.model.backbone.eagle_model.language_model.config.output_attentions = True
        
        # Language Model layers
        llm_layers = self.model.backbone.eagle_model.language_model.model.layers
        for idx, layer in enumerate(llm_layers):
            hook_name = f"llm.layer_{idx}"
            hook = layer.self_attn.register_forward_hook(
                self._create_attention_hook(hook_name)
            )
            self.hooks.append(hook)
            print(f"Registered attention hook: {hook_name}")
        
        print(f"\nTotal hooks registered: {len(self.hooks)}")
        return self
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_attentions(self) -> Dict[str, torch.Tensor]:
        """
        Get the extracted attention weights.
        
        Returns:
            Dict mapping layer names to attention weight tensors.
            Keys: "llm.layer_{idx}.attn_step_{step}"
            Shape: (batch_size, num_heads, seq_len, seq_len)
        """
        return self.attentions.copy()
    
    def clear_attentions(self):
        """Clear stored attention weights to free memory."""
        self.attentions.clear()
        self.step_counters.clear()
    
    def __enter__(self):
        """Context manager entry - registers hooks."""
        self.register_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - removes hooks."""
        self.clear_hooks()
        self.clear_attentions()



def extract_features_from_forward(model, inputs: dict) -> Dict[str, torch.Tensor]:
    """
    Convenience function to extract features during a single forward pass.
    
    Args:
        model: GR00T_N1_5 model instance
        inputs: Input dictionary for the model
        extractor: (Optional) Pre-initialized IntermediateFeatureExtractor instance.
                   If None, creates a new one (less efficient for repeated calls).
        
    Returns:
        Dictionary of intermediate features from all transformer layers
        
    Example (one-time extraction):
        ```python
        from feature_extraction_utils import extract_features_from_forward
        
        # Run model and extract features
        features = extract_features_from_forward(model, inputs)
        
        # Access features
        for layer_name, feature_tensor in features.items():
            print(f"{layer_name}: {feature_tensor.shape}")
        ```
    
    Example (repeated extraction - more efficient):
        ```python
        from feature_extraction_utils import IntermediateFeatureExtractor, extract_features_from_forward
        
        # Create extractor once and reuse
        extractor = IntermediateFeatureExtractor(model)
        extractor.register_hooks()  # Register hooks once
        
        for batch in dataloader:
            # Reuse extractor for each batch (more efficient)
            features = extract_features_from_forward(model, batch, extractor=extractor)
            # Process features...
            extractor.clear_features()  # Clear to free memory
        
        extractor.clear_hooks()  # Clean up when done
        ```
    """
    extractor = LLMAttentionExtractor(model)
    
    with extractor:
        # Run forward pass
        _ = model.get_action(inputs)
        # Get extracted features
        features = extractor.get_attentions()
    
    return features



# load model and processor
model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path="nvidia/GR00T-N1.5-3B",
        tune_llm=True,  # backbone's LLM
        tune_visual=True,  # backbone's vision tower
        tune_projector=False,  # action head's projector
        tune_diffusion_model=False,  # action head's DiT
        torch_dtype=torch.bfloat16,
    )
model = model.to("cuda")
model.eval()  # Set to evaluation mode (disable dropout, etc.)
    
# Get EAGLE VLM backbone
#eagle_model = model.backbone.eagle_mode

# get dataset
data_config_cls = DATA_CONFIG_MAP["bimanual_piper"]
modality_configs = data_config_cls.modality_config()
transforms = data_config_cls.transform()
embodiment_tag = EmbodimentTag("new_embodiment")

train_dataset = LeRobotSingleDataset(
    dataset_path="/home/dongjun/gr00t_svcca/dataset/segmented_depth_1_trash_separation_3_objs_unseen_251126_pnp_coke_to_plastic_place_split",
    modality_configs=modality_configs,
    transforms=transforms,
    embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
    video_backend="torchvision_av",
    )

  # Get eagle processor
eagle_processor = build_eagle_processor(
'/home/dongjun/GR00T_Training/gr00t/model/backbone/eagle2_hg_model',
)
eagle_processor.tokenizer.padding_side = "left"

sample = next(iter(train_dataset))

inputs = collate([sample], eagle_processor)

out = extract_features_from_forward(model, inputs)