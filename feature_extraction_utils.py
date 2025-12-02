# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility module for extracting intermediate features from GR00T model.
This module provides hooks to extract features from all transformer layers in:
1. Eagle VLM Backbone (vision tower + LLM layers)
2. DiT Action Head (transformer blocks)
"""

import torch
from typing import Dict, List
from collections import OrderedDict


class IntermediateFeatureExtractor:
    """
    A hook-based feature extractor for GR00T model.
    Extracts intermediate features from all transformer layers in the model.
    """
    
    def __init__(self, model):
        """
        Args:
            model: GR00T_N1_5 model instance
        """
        self.model = model
        self.features = OrderedDict()
        self.hooks = []
        self.step_counters = {}  # Track step count for each layer (for diffusion)
        
    def _create_hook(self, name: str):
        """Create a forward hook that stores the output tensor with step tracking."""
        def hook(module, input, output):
            # Initialize step counter for this layer if not exists
            if name not in self.step_counters:
                self.step_counters[name] = 0
            
            # Create unique name with step index
            step_name = f"{name}_step_{self.step_counters[name]}"
            
            # Store the output feature
            if isinstance(output, torch.Tensor):
                self.features[step_name] = output.detach()
            elif isinstance(output, tuple) and len(output) > 0:
                # Some modules return tuples, take the first element (usually hidden states)
                self.features[step_name] = output[0].detach() if isinstance(output[0], torch.Tensor) else output[0]
            else:
                self.features[step_name] = output
            
            # Increment step counter
            self.step_counters[name] += 1
        return hook
    
    def register_hooks(self):
        """Register forward hooks on all transformer layers."""
        self.clear_hooks()
        self.features.clear()
        
        # 1. Language Model - Last layer only (Qwen3DecoderLayer)
        llm_layers = self.model.backbone.eagle_model.language_model.model.layers
        last_llm_idx = len(llm_layers) - 1
        hook_name = f"backbone.llm.layer_{last_llm_idx}"
        hook = llm_layers[last_llm_idx].register_forward_hook(self._create_hook(hook_name))
        self.hooks.append(hook)
        print(f"Registered hook: {hook_name}")
        
        # 2. Action Head State Encoder - CategorySpecificMLP (layer1, layer2)
        hook_name = "action_head.state_encoder.layer1"
        hook = self.model.action_head.state_encoder.layer1.register_forward_hook(
            self._create_hook(hook_name)
        )
        self.hooks.append(hook)
        print(f"Registered hook: {hook_name}")
        
        hook_name = "action_head.state_encoder.layer2"
        hook = self.model.action_head.state_encoder.layer2.register_forward_hook(
            self._create_hook(hook_name)
        )
        self.hooks.append(hook)
        print(f"Registered hook: {hook_name}")
        
        # 3. Action Head VL Self-Attention - SelfAttentionTransformer blocks
        vl_attn_blocks = self.model.action_head.vl_self_attention.transformer_blocks
        for idx, block in enumerate(vl_attn_blocks):
            hook_name = f"action_head.vl_self_attention.layer_{idx}"
            hook = block.register_forward_hook(self._create_hook(hook_name))
            self.hooks.append(hook)
            print(f"Registered hook: {hook_name}")
        
        # 4. Action Head DiT - 12 x BasicTransformerBlock
        #dit_blocks = self.model.action_head.model.transformer_blocks
        #for idx, block in enumerate(dit_blocks):
        #    hook_name = f"action_head.dit.layer_{idx}"
        #    hook = block.register_forward_hook(self._create_hook(hook_name))
        #    self.hooks.append(hook)
        #    print(f"Registered hook: {hook_name}")
        
        print(f"\nTotal hooks registered: {len(self.hooks)}")
        return self
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_features(self) -> Dict[str, torch.Tensor]:
        """
        Get the extracted intermediate features.
        
        Returns:
            Dict mapping layer names to their output tensors.
            Keys follow the format:
            - "backbone.vision.layer_{idx}" for vision tower layers
            - "backbone.llm.layer_{idx}" for LLM layers
            - "action_head.vl_self_attention.layer_{idx}" for VL self-attention layers
            - "action_head.dit.layer_{idx}" for DiT layers
        """
        return self.features.copy()
    
    def clear_features(self):
        """Clear stored features to free memory."""
        self.features.clear()
        self.step_counters.clear()  # Reset step counters when clearing features
    
    def __enter__(self):
        """Context manager entry - registers hooks."""
        self.register_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - removes hooks."""
        self.clear_hooks()
        self.clear_features()


def extract_features_from_forward(model, inputs: dict, extractor=None) -> Dict[str, torch.Tensor]:
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
    # If extractor is provided, use it (assumes hooks are already registered)
    if extractor is not None:
        # Clear previous features
        extractor.clear_features()
        # Run forward pass
        _ = model.get_action(inputs)
        # Get extracted features
        features = extractor.get_features()
        return features
    
    # Otherwise, create temporary extractor (less efficient for repeated calls)
    else:
        temp_extractor = IntermediateFeatureExtractor(model)
        with temp_extractor:
            # Run forward pass
            _ = model.get_action(inputs)
            # Get extracted features
            features = temp_extractor.get_features()
        return features


def print_feature_shapes(features: Dict[str, torch.Tensor]):
    """
    Utility function to print shapes of all extracted features.
    
    Args:
        features: Dictionary of features from extract_features_from_forward
    """
    print("\n" + "="*80)
    print("EXTRACTED INTERMEDIATE FEATURES")
    print("="*80)
    
    # Group by module
    backbone_vision = {k: v for k, v in features.items() if k.startswith("backbone.vision")}
    backbone_llm = {k: v for k, v in features.items() if k.startswith("backbone.llm")}
    ah_vl_attn = {k: v for k, v in features.items() if k.startswith("action_head.vl_self_attention")}
    ah_dit = {k: v for k, v in features.items() if k.startswith("action_head.dit")}
    
    if backbone_vision:
        print("\n[Backbone - Vision Tower]")
        for name, tensor in backbone_vision.items():
            print(f"  {name}: {tensor.shape} [{tensor.dtype}]")
    
    if backbone_llm:
        print("\n[Backbone - LLM Layers]")
        for name, tensor in backbone_llm.items():
            print(f"  {name}: {tensor.shape} [{tensor.dtype}]")
    
    if ah_vl_attn:
        print("\n[Action Head - VL Self-Attention]")
        for name, tensor in ah_vl_attn.items():
            print(f"  {name}: {tensor.shape} [{tensor.dtype}]")
    
    if ah_dit:
        print("\n[Action Head - DiT Transformer]")
        for name, tensor in ah_dit.items():
            print(f"  {name}: {tensor.shape} [{tensor.dtype}]")
    
    print("\n" + "="*80)
    print(f"Total layers: {len(features)}")
    print("="*80 + "\n")

