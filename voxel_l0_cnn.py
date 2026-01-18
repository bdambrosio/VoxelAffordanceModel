"""
voxel_l0_cnn.py

3×3×3 local-context voxel → L0 perceptual evidence model.

- Fixed receptive field
- Strong geometric inductive bias
- Predicts labels for CENTER voxel only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any


class PerceptionModel3DCNN(nn.Module):
    """
    3D CNN model that processes a 3×3×3 neighborhood of voxels.
    
    Input: [B, C, 3, 3, 3] where C = feature channels per voxel
    Output: Predictions for center voxel only [B, ...]
    
    Feature channels per voxel (13 total):
    - occupancy: 1 (int 0-3, normalized)
    - material_class: 1 (int 0-6, normalized)
    - hazard_class: 1 (int 0-2, normalized)
    - breakability: 1 (int 0-1)
    - tool_hint: 1 (int 0-4, normalized)
    - dx, dy, dz: 3 (normalized)
    - age_s: 1 (normalized)
    - support, passable, head_obstacle, thin_layer: 4 (0/1)
    """
    
    def __init__(
        self,
        in_channels: int = 13,          # per-voxel feature depth
        hidden_channels: int = 32,

        num_structure_types: int = 8,
        num_affordance_types: int = 8,
        num_risk_types: int = 4,
        num_severity: int = 3,
    ):
        super().__init__()

        # -------------------------
        # 3D Conv trunk
        # -------------------------

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Pool to center voxel (implicit via indexing)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
        )

        # -------------------------
        # Heads (center voxel only)
        # -------------------------

        self.structure_head = nn.Linear(128, num_structure_types)
        self.affordance_head = nn.Linear(
            128, num_affordance_types  # multi-label logits (sigmoid at use-site)
        )
        self.risk_head = nn.Linear(
            128, num_risk_types + (num_risk_types * num_severity)
        )

    def forward(self, x):
        """
        x: [B, C, 3, 3, 3]
        """

        h = self.conv(x)

        # Extract CENTER voxel features
        center = h[:, :, 1, 1, 1]

        z = self.fc(center)

        return {
            "structure": self.structure_head(z),   # logits
            "affordance": self.affordance_head(z), # logits
            "risk": self.risk_head(z),             # logits (split at use-site)
        }


def load_model(model_path: str = "models/voxel_l0_model", device: str = "cuda") -> PerceptionModel3DCNN:
    """
    Load a trained PerceptionModel3DCNN from disk.
    
    Args:
        model_path: Path to saved model directory (default: models/voxel_l0_model)
        device: Device to load model on ("cuda" or "cpu")
        
    Returns:
        Loaded PerceptionModel3DCNN in eval mode
    """
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path_obj}")
    
    model = PerceptionModel3DCNN()
    
    # Load state dict - try safetensors first (used by transformers), then pytorch_model.bin
    safetensors_path = model_path_obj / "model.safetensors"
    state_dict_path = model_path_obj / "pytorch_model.bin"
    
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
            model.load_state_dict(state_dict)
        except ImportError:
            # Fall back to pytorch format if safetensors not available
            if state_dict_path.exists():
                model.load_state_dict(torch.load(state_dict_path, map_location=device))
            else:
                raise FileNotFoundError(
                    f"Safetensors file found but safetensors library not installed. "
                    f"Install with: pip install safetensors"
                )
    elif state_dict_path.exists():
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
    else:
        raise FileNotFoundError(
            f"No model file found in {model_path_obj}. "
            f"Expected either 'model.safetensors' or 'pytorch_model.bin'"
        )
    
    model = model.to(device)
    model.eval()
    
    return model
