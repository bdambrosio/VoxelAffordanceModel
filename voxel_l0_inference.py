"""
voxel_l0_inference.py

User-facing inference pipeline for trained PerceptionModel3DCNN.

Takes raw partial grid observations and returns PerceptionFrame.
"""

import torch
import time
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from voxel_l0_cnn import PerceptionModel3DCNN, load_model
from canonicalize import Canonicalize
from tokens import TokenBuilder
from perceptual_frame import (
    PerceptionFrame,
    Structure,
    Affordance,
    Risk,
    StructureType,
    AffordanceType,
    RiskType,
    Severity,
    PerceptualAssembler,
    AFFORDANCE_TYPES,
    RISK_TYPES,
    SEVERITIES,
)

SEVERITY_REVERSE = {
    0: Severity.LOW,
    1: Severity.MEDIUM,
    2: Severity.HIGH,
}


class InferencePipeline:
    """
    Inference pipeline that loads model once and processes partial grid observations.
    
    Usage:
        pipeline = InferencePipeline(model_path="models/voxel_l0_model")
        frame = pipeline.infer_partial_grid(raw_partial_grid)
    """
    
    def __init__(
        self,
        model_path: str = "models/voxel_l0_model",
        device: str = "cuda",
        batch_size: int = 64
    ):
        """
        Initialize pipeline and load model.
        
        Args:
            model_path: Path to saved model directory
            device: Device to run inference on ("cuda" or "cpu")
            batch_size: Batch size for processing multiple voxels
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.batch_size = batch_size
        
        # Load model once at startup
        self.model = load_model(model_path, device=self.device)
        self.canon = Canonicalize()
        self.token_builder = TokenBuilder()
        self.assembler = PerceptualAssembler()
        
        print(f"InferencePipeline initialized: model loaded from {model_path}, device={self.device}")
    
    def _extract_neighborhood(
        self,
        center_pos: Tuple[int, int, int],
        partial_canon: Dict[str, Any],
        agent_pos: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Extract 3×3×3 neighborhood around center_pos from partial canonicalization.
        
        Args:
            center_pos: (x, y, z) absolute position of center voxel
            partial_canon: Canonicalized partial grid dict
            agent_pos: (x, y, z) agent position
            
        Returns:
            Feature tensor [13, 3, 3, 3]
        """
        # Build position lookup
        pos_to_cell = {tuple(cell["pos"]): cell for cell in partial_canon["cells"]}
        
        # Fixed order: iterate y (low to high), then x, then z
        features = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_pos = (center_pos[0] + dx, center_pos[1] + dy, center_pos[2] + dz)
                    cell = pos_to_cell.get(neighbor_pos)
                    
                    if cell:
                        token = self.token_builder.from_canonical_cell(cell, agent_pos)
                        # Normalize features: ints to [0,1], bools to 0/1
                        feat = torch.tensor([
                            float(token.occupancy.value) / 3.0,  # normalize 0-3
                            float(token.material_class.value) / 6.0,  # normalize 0-6
                            float(token.hazard_class.value) / 2.0,  # normalize 0-2
                            float(token.breakability.value),  # already 0-1
                            float(token.tool_hint.value) / 4.0,  # normalize 0-4
                            float(token.dx) / 50.0,  # normalize spatial (assume max ~50)
                            float(token.dy) / 50.0,
                            float(token.dz) / 50.0,
                            float(token.age_s) / 1000.0,  # normalize age (assume max ~1000s)
                            float(token.support),
                            float(token.passable),
                            float(token.head_obstacle),
                            float(token.thin_layer),
                        ], dtype=torch.float32)
                    else:
                        # Zero-pad missing neighbors
                        feat = torch.zeros(13, dtype=torch.float32)
                    
                    features.append(feat)
        
        # Stack into [3, 3, 3, 13] then permute to [13, 3, 3, 3]
        features_tensor = torch.stack(features).reshape(3, 3, 3, 13).permute(3, 0, 1, 2)
        return features_tensor
    
    def _model_outputs_to_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        relative_positions: List[Tuple[int, int, int]]
    ) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
        """
        Convert model outputs to label format compatible with PerceptionFrame.
        
        Args:
            outputs: Model outputs dict with structure [B, |StructureType|], affordance [B, |AffordanceType|], risk [B, 10]
            relative_positions: List of (dx, dy, dz) relative positions for each output
            
        Returns:
            Dict mapping (dx, dy, dz) -> label dict with structure_type_scores, affordances, risks
        """
        structure_logits = outputs["structure"].cpu()  # [B, |StructureType|]
        affordance_logits = outputs["affordance"].cpu()  # [B, |AffordanceType|]
        risk = outputs["risk"].cpu()  # [B, |RiskType| + |RiskType|*3]
        
        labels = {}
        
        for i, rel_pos in enumerate(relative_positions):
            # Structure types: sigmoid over learned logits
            st_probs = torch.sigmoid(structure_logits[i])
            structure_scores = {
                st: float(st_probs[j]) for j, st in enumerate(StructureType)
            }
            
            # Affordances: multi-label sigmoid + threshold
            aff_probs = torch.sigmoid(affordance_logits[i])
            affordances = [
                AFFORDANCE_TYPES[j].value
                for j in range(len(AFFORDANCE_TYPES))
                if float(aff_probs[j]) > 0.5
            ]
            
            # Risks: multi-label presence + per-type severity
            num_risk_types = len(RISK_TYPES)
            risk_present_logits = risk[i, :num_risk_types]
            risk_sev_logits = risk[i, num_risk_types:].reshape(num_risk_types, 3)
            
            risk_present = torch.sigmoid(risk_present_logits)
            
            risks = []
            for j, rtype in enumerate(RISK_TYPES):
                if float(risk_present[j]) <= 0.5:
                    continue
                sev_idx = int(torch.argmax(risk_sev_logits[j]))
                severity = SEVERITY_REVERSE.get(sev_idx, Severity.LOW)
                risks.append({"type": rtype.value, "severity": severity.value})
            
            labels[rel_pos] = {
                "structure_type_scores": {k.value: v for k, v in structure_scores.items()},
                "affordances": affordances,
                "risks": risks,
            }
        
        return labels
    
    def infer_partial_grid(
        self,
        raw_partial_grid: Dict[str, Any],
        boundary: Optional[int] = None
    ) -> PerceptionFrame:
        """
        Run inference on a raw partial grid observation.
        
        Args:
            raw_partial_grid: Raw partial grid dict with structure:
                {
                    "radius": int,
                    "center": {"x": int, "y": int, "z": int},
                    "cells": {
                        "x,y,z": {"name": "minecraft:block", "ts": float},
                        ...
                    }
                }
            boundary: Optional boundary parameter. If provided:
                - Inference is performed only on voxels within +/- (boundary+1) for dx, dy, dz centered on 'center'
                - PerceptionFrame is reported only for voxels within +/- boundary
        
        Returns:
            PerceptionFrame with structures, affordances, and risks
        """
        # Canonicalize partial grid
        partial_canon = self.canon.canonicalize(raw_partial_grid)
        center = partial_canon["center"]
        agent_pos = center  # Agent is at center
        
        # Check if we have any cells
        if not partial_canon.get("cells"):
            # Empty grid - return empty frame
            return PerceptionFrame(timestamp=time.time())
        
        # Extract neighborhoods for voxels (filtered by boundary if provided)
        neighborhoods = []
        relative_positions = []
        
        for cell in partial_canon["cells"]:
            center_pos = cell["pos"]
            rel_pos = (
                center_pos[0] - agent_pos[0],
                center_pos[1] - agent_pos[1],
                center_pos[2] - agent_pos[2]
            )
            
            # If boundary is provided, only process voxels within +/- (boundary+1)
            if boundary is not None:
                if (abs(rel_pos[0]) > boundary + 1 or 
                    abs(rel_pos[1]) > boundary + 1 or 
                    abs(rel_pos[2]) > boundary + 1):
                    continue
            
            neighborhood = self._extract_neighborhood(center_pos, partial_canon, agent_pos)
            neighborhoods.append(neighborhood)
            relative_positions.append(rel_pos)
        
        if not neighborhoods:
            # Empty grid - return empty frame
            return PerceptionFrame(timestamp=time.time())
        
        # Batch inference
        all_outputs = {
            "structure": [],
            "affordance": [],
            "risk": [],
        }
        
        self.model.eval()
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(neighborhoods), self.batch_size):
                batch_neighborhoods = neighborhoods[i:i + self.batch_size]
                batch_tensor = torch.stack(batch_neighborhoods).to(self.device)  # [B, 13, 3, 3, 3]
                
                batch_outputs = self.model(batch_tensor)
                
                all_outputs["structure"].append(batch_outputs["structure"])
                all_outputs["affordance"].append(batch_outputs["affordance"])
                all_outputs["risk"].append(batch_outputs["risk"])
        
        # Concatenate all batches
        outputs = {
            "structure": torch.cat(all_outputs["structure"], dim=0),
            "affordance": torch.cat(all_outputs["affordance"], dim=0),
            "risk": torch.cat(all_outputs["risk"], dim=0),
        }
        
        # Convert to labels
        voxel_labels = self._model_outputs_to_labels(outputs, relative_positions)
        
        # If boundary is provided, filter labels to only include voxels within +/- boundary
        if boundary is not None:
            filtered_voxel_labels = {}
            for rel_pos, labels in voxel_labels.items():
                if (abs(rel_pos[0]) <= boundary and 
                    abs(rel_pos[1]) <= boundary and 
                    abs(rel_pos[2]) <= boundary):
                    filtered_voxel_labels[rel_pos] = labels
            voxel_labels = filtered_voxel_labels
        
        # Store per-voxel labels for reporting
        self._last_voxel_labels = voxel_labels
        
        # Use PerceptualAssembler to group voxels into structures
        frame = self.assembler.assemble(voxel_labels)
        
        return frame
    
    def get_per_voxel_stats(self) -> Dict[str, Any]:
        """
        Get statistics on per-voxel predictions (before assembly).
        
        Returns:
            Dict with counts by type
        """
        if not hasattr(self, '_last_voxel_labels'):
            return {}
        
        structure_counts = {}
        affordance_counts = {}
        risk_counts = {}
        
        for labels in self._last_voxel_labels.values():
            # Count structure types (dominant type per voxel)
            scores = labels["structure_type_scores"]
            dominant_type = max(scores.items(), key=lambda x: x[1])[0]
            structure_counts[dominant_type] = structure_counts.get(dominant_type, 0) + 1
            
            # Count affordances
            for aff_type_str in labels["affordances"]:
                affordance_counts[aff_type_str] = affordance_counts.get(aff_type_str, 0) + 1
            
            # Count risks
            for risk_dict in labels["risks"]:
                key = (risk_dict["type"], risk_dict["severity"])
                risk_counts[key] = risk_counts.get(key, 0) + 1
        
        return {
            "structures": structure_counts,
            "affordances": affordance_counts,
            "risks": risk_counts,
        }


def main():
    """
    Example usage.
    """
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference on a raw partial grid")
    parser.add_argument(
        "input_file",
        type=str,
        nargs="?",
        default=None,
        help="Path to JSON file containing raw partial grid (default: first file in training_data directory)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/voxel_l0_model",
        help="Path to saved model directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--boundary",
        type=int,
        default=1,
        help="Boundary parameter: inference on voxels within +/- (boundary+1), report on voxels within +/- boundary"
    )
    
    args = parser.parse_args()
    
    # If no input file provided, use first file in training_data directory
    if args.input_file is None:
        training_data_dir = Path("training_data")
        if not training_data_dir.exists():
            parser.error("No input file provided and training_data directory not found")
        
        json_files = sorted(training_data_dir.glob("*.json"))
        if not json_files:
            parser.error("No JSON files found in training_data directory")
        
        args.input_file = str(json_files[0])
        print(f"No input file provided, using first file from training_data: {args.input_file}")
    
    # Load raw partial grid
    with open(args.input_file) as f:
        data = json.load(f)
    
    # Extract partial_grid from record (training data format)
    if "partial_grid" in data:
        raw_partial_grid = data["partial_grid"]
    else:
        # Assume it's already a partial_grid dict
        raw_partial_grid = data
    
    # Initialize pipeline (loads model once)
    pipeline = InferencePipeline(model_path=args.model_path, device=args.device)
    
    # Run inference
    frame = pipeline.infer_partial_grid(raw_partial_grid, boundary=args.boundary)
    
    # Count voxels processed
    if "partial_grid" in data:
        num_voxels = len(data["partial_grid"].get("cells", {}))
    else:
        num_voxels = len(data.get("cells", {}))
    
    # Get per-voxel statistics
    per_voxel_stats = pipeline.get_per_voxel_stats()
    
    # Print per-voxel stage
    print(f"\n{'='*60}")
    print(f"STAGE 1: Per-Voxel Predictions")
    print(f"{'='*60}")
    print(f"Voxels processed: {num_voxels}")
    
    if per_voxel_stats:
        print(f"\nPer-Voxel Structures: {sum(per_voxel_stats['structures'].values())}")
        for struct_type, count in sorted(per_voxel_stats['structures'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {struct_type}: {count}")
        
        print(f"\nPer-Voxel Affordances: {sum(per_voxel_stats['affordances'].values())}")
        for aff_type, count in sorted(per_voxel_stats['affordances'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {aff_type}: {count}")
        
        print(f"\nPer-Voxel Risks: {sum(per_voxel_stats['risks'].values())}")
        for (risk_type, severity), count in sorted(per_voxel_stats['risks'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {risk_type} ({severity}): {count}")
    
    # Count assembled structures by type
    structure_counts = {}
    for struct in frame.structures:
        dominant_type = struct.dominant_type()
        # dominant_type() returns StructureType enum, but type_scores uses string keys
        # So we need to handle both cases
        if isinstance(dominant_type, str):
            struct_key = dominant_type
        else:
            struct_key = dominant_type.value
        structure_counts[struct_key] = structure_counts.get(struct_key, 0) + 1
    
    affordance_counts = {}
    for aff in frame.affordances:
        aff_key = aff.type.value if hasattr(aff.type, 'value') else str(aff.type)
        affordance_counts[aff_key] = affordance_counts.get(aff_key, 0) + 1
    
    risk_counts = {}
    for risk in frame.risks:
        risk_key = risk.type.value if hasattr(risk.type, 'value') else str(risk.type)
        severity_key = risk.severity.value if hasattr(risk.severity, 'value') else str(risk.severity)
        key = (risk_key, severity_key)
        risk_counts[key] = risk_counts.get(key, 0) + 1
    
    # Print assembled stage
    print(f"\n{'='*60}")
    print(f"STAGE 2: Assembled PerceptionFrame")
    print(f"{'='*60}")
    print(frame.summary())
    print(f"\nAssembled Structures: {len(frame.structures)}")
    for struct_type, count in sorted(structure_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {struct_type}: {count}")
    
    print(f"\nAssembled Affordances: {len(frame.affordances)}")
    for aff_type, count in sorted(affordance_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {aff_type}: {count}")
    
    print(f"\nAssembled Risks: {len(frame.risks)}")
    for (risk_type, severity), count in sorted(risk_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {risk_type} ({severity}): {count}")


if __name__ == "__main__":
    main()
