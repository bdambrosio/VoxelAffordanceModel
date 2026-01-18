"""
training_harness.py

Training harness for PerceptionModel.
Processes training records, canonicalizes data, and trains the model.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from voxel_l0_cnn import PerceptionModel3DCNN
from canonicalize import Canonicalize
from tokens import TokenBuilder
from ground_truth_labels import get_ground_truth_labels
from perceptual_frame import (
    AffordanceType,
    RiskType,
    Severity,
    StructureType,
    AFFORDANCE_TYPES,
    RISK_TYPES,
    SEVERITIES,
)

# Configure logging with console output
# Use basicConfig to ensure console output, but allow override
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],  # Explicitly add console handler
    force=True  # Override any previous basicConfig calls
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ensure root logger also has console handler
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Add console handler if not present
if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

# Constants
STRUCTURE_TYPES: List[StructureType] = list(StructureType)
SEVERITY_TO_INDEX = {sev: i for i, sev in enumerate(SEVERITIES)}


# -------------------------------------------------
# Dataset
# -------------------------------------------------

class PerceptionDataset(Dataset):
    """
    Dataset for perception model training.
    
    Each sample contains batch (model inputs) and targets (ground truth labels).
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        """
        Initialize dataset.
        
        Args:
            samples: List of training samples with batch and targets dicts
        """
        if not samples:
            raise ValueError("Dataset cannot be empty")
        self.samples = samples
        logger.info(f"Initialized PerceptionDataset with {len(samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range [0, {len(self.samples)})")
        sample = self.samples[idx]
        # Validate sample structure
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {idx} is not a dict: {type(sample)}")
        if "batch" not in sample:
            raise ValueError(f"Sample {idx} missing 'batch' key. Keys: {list(sample.keys())}")
        if "targets" not in sample:
            # Log the actual sample to help debug
            logger.error(f"Sample {idx} in dataset missing 'targets'. Keys: {list(sample.keys())}")
            logger.error(f"Sample type: {type(sample)}")
            logger.error(f"Sample repr (first 500 chars): {repr(sample)[:500]}")
            # Check if it's a different structure
            if isinstance(sample.get("batch"), dict) and "targets" in sample.get("batch", {}):
                logger.error("Found 'targets' nested inside 'batch' - possible structure mismatch")
            raise ValueError(
                f"Sample {idx} missing 'targets' key. Keys: {list(sample.keys())}. "
                f"Sample type: {type(sample)}, Sample repr: {repr(sample)[:200]}"
            )
        # Make a copy to ensure we don't accidentally modify the original
        result = {"batch": sample["batch"], "targets": sample["targets"]}
        # Double-check the copy
        if "targets" not in result:
            raise ValueError(f"Copy of sample {idx} missing 'targets' after copy operation")
        return result


def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching samples.
    
    Args:
        samples: List of sample dicts
        
    Returns:
        Batched dict with "batch" and "targets" keys
        Note: Trainer automatically moves tensors to the correct device
    """
    collate_start = time.time()
    
    if not samples:
        raise ValueError("Cannot collate empty batch")
    
    # Validate sample structure
    if "batch" not in samples[0]:
        raise ValueError(
            f"Sample 0 missing 'batch' key. Sample keys: {list(samples[0].keys())}. "
            f"Sample type: {type(samples[0])}, Sample repr: {repr(samples[0])[:200]}"
        )
    if "targets" not in samples[0]:
        raise ValueError(
            f"Sample 0 missing 'targets' key. Sample keys: {list(samples[0].keys())}. "
            f"Sample type: {type(samples[0])}, Sample repr: {repr(samples[0])[:200]}. "
            f"All samples in batch: {[list(s.keys()) for s in samples]}"
        )
    
    batch = {}
    targets = {}
    
    # Validate all samples have same keys
    first_batch_keys = set(samples[0]["batch"].keys())
    first_target_keys = set(samples[0]["targets"].keys())
    
    for i, s in enumerate(samples):
        if set(s["batch"].keys()) != first_batch_keys:
            raise ValueError(f"Sample {i} has different batch keys")
        if set(s["targets"].keys()) != first_target_keys:
            raise ValueError(f"Sample {i} has different target keys")
    
    # Stack batch tensors
    stack_start = time.time()
    for key in samples[0]["batch"]:
        batch[key] = torch.stack([s["batch"][key] for s in samples])  # [B, 13, 3, 3, 3]
    
    # Stack target tensors
    for key in samples[0]["targets"]:
        targets[key] = torch.stack([s["targets"][key] for s in samples])  # [B, ...]
    stack_time = time.time() - stack_start
    
    collate_time = time.time() - collate_start
    
    # Log collation time if significant (only log occasionally to avoid spam)
    if collate_time > 0.1:  # Only log if it takes significant time
        logger.debug(f"Collation time: {collate_time*1000:.2f}ms (stack: {stack_time*1000:.2f}ms) for {len(samples)} samples")
    
    # Note: Trainer will move tensors to device automatically
    return {"batch": batch, "targets": targets}


# -------------------------------------------------
# Loss
# -------------------------------------------------

def compute_loss(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute multi-task loss for perception model.
    
    Args:
        outputs: Model outputs dict with structure [B, |StructureType|], affordance [B, |AffordanceType|], risk [B, 10]
        targets: Target dict with all ground truth labels [B, ...]
        
    Returns:
        Total loss tensor
    """
    mask = targets["valid_mask"].float()  # [B]
    mask_sum = mask.sum()
    
    # Guard against division by zero
    if mask_sum == 0:
        logger.warning("Empty batch (all invalid), returning zero loss")
        return torch.tensor(0.0, device=mask.device, requires_grad=True)
    
    # -------------------------
    # Structure types (multi-label)
    # -------------------------
    struct_loss = F.binary_cross_entropy_with_logits(
        outputs["structure"],
        targets["structure_type_scores"],
        reduction="none"
    ).mean(dim=-1)  # [B]
    
    struct_loss = (struct_loss * mask).sum() / mask_sum
    
    # -------------------------
    # Affordances (multi-label)
    # -------------------------
    aff_loss = F.binary_cross_entropy_with_logits(
        outputs["affordance"],
        targets["affordance_multi"],
        reduction="none"
    ).mean(dim=-1)  # [B]
    
    aff_loss = (aff_loss * mask).sum() / mask_sum
    
    # -------------------------
    # Risks (multi-label + per-type severity)
    # -------------------------
    risk_out = outputs["risk"]  # [B, |RiskType| + |RiskType|*3]
    num_risk_types = len(RISK_TYPES)
    num_severity = len(SEVERITIES)  # 3
    
    risk_present_logits = risk_out[:, :num_risk_types]  # [B, T]
    risk_sev_logits = risk_out[:, num_risk_types:].reshape(-1, num_risk_types, num_severity)  # [B, T, 3]
    
    risk_present_loss = F.binary_cross_entropy_with_logits(
        risk_present_logits,
        targets["risk_present"],
        reduction="none"
    ).mean(dim=-1)  # [B]
    
    # Severity loss only for types that are present; absent types are ignore_index
    risk_sev_loss = F.cross_entropy(
        risk_sev_logits.reshape(-1, num_severity),                    # [B*T, 3]
        targets["risk_severity_by_type"].reshape(-1),                 # [B*T]
        reduction="none",
        ignore_index=-100
    ).reshape(-1, num_risk_types).mean(dim=-1)  # [B]
    
    risk_loss = ((risk_present_loss + risk_sev_loss) * mask).sum() / mask_sum
    
    # -------------------------
    # Total
    # -------------------------
    total_loss = struct_loss + aff_loss + risk_loss
    
    return total_loss


# -------------------------------------------------
# Trainer
# -------------------------------------------------

class PerceptionTrainer(Trainer):
    """
    Custom trainer with loss component logging and timing.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_times = []
        self.forward_times = []
        self.loss_times = []
        self.backward_times = []
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override to add timing for each training step.
        """
        step_start = time.time()
        
        # Forward pass
        forward_start = time.time()
        # Newer transformers passes num_items_in_batch; keep compatibility with older versions.
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        forward_time = time.time() - forward_start
        
        step_time = time.time() - step_start
        
        # Track timing
        self.forward_times.append(forward_time)
        self.step_times.append(step_time)
        
        # Log timing periodically
        if self.state.global_step % self.args.logging_steps == 0 and len(self.step_times) > 0:
            avg_step_time = sum(self.step_times[-self.args.logging_steps:]) / len(self.step_times[-self.args.logging_steps:])
            avg_forward_time = sum(self.forward_times[-self.args.logging_steps:]) / len(self.forward_times[-self.args.logging_steps:])
            self.log({
                "train/step_time_ms": avg_step_time * 1000,
                "train/forward_time_ms": avg_forward_time * 1000,
            })
        
        return loss
    
    def compute_loss(
        self, 
        model, 
        inputs: Dict[str, Any], 
        return_outputs: bool = False, 
        num_items_in_batch: Optional[int] = None
    ):
        """
        Compute loss and optionally log components.
        
        Args:
            model: The model to compute loss for
            inputs: Input dict with 'batch' and 'targets' keys
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (for compatibility with newer transformers)
        """
        forward_start = time.time()
        # Model expects [B, C, 3, 3, 3] input
        neighborhood = inputs["batch"]["neighborhood"]  # [B, 13, 3, 3, 3]
        outputs = model(neighborhood)
        forward_time = time.time() - forward_start
        
        loss_start = time.time()
        # Type assertion: inputs["targets"] is Dict[str, torch.Tensor] based on our data structure
        targets: Dict[str, torch.Tensor] = inputs["targets"]  # type: ignore[assignment]
        loss = compute_loss(outputs, targets)
        loss_time = time.time() - loss_start
        
        # Log individual loss components periodically
        if self.state.global_step % self.args.logging_steps == 0:
            mask = inputs["targets"]["valid_mask"].float()
            mask_sum = mask.sum()
            if mask_sum > 0:
                # Compute components for logging
                struct_loss = F.binary_cross_entropy_with_logits(
                    outputs["structure"],
                    inputs["targets"]["structure_type_scores"],
                    reduction="none"
                ).mean(dim=-1)  # [B]
                struct_loss = (struct_loss * mask).sum() / mask_sum
                
                # Approximate other components
                total_loss_val = loss.item()
                struct_loss_val = struct_loss.item()
                other_loss = total_loss_val - struct_loss_val
                
                self.log({
                    "train/loss": total_loss_val,
                    "train/loss_structure": struct_loss_val,
                    "train/loss_affordance": other_loss * 0.5,
                    "train/loss_risk": other_loss * 0.5,
                    "train/forward_time_ms": forward_time * 1000,
                    "train/loss_compute_time_ms": loss_time * 1000,
                })
        
        if return_outputs:
            return (loss, outputs)  # type: ignore[return-value]
        return loss


# -------------------------------------------------
# Data Processing
# -------------------------------------------------

def get_3x3x3_neighborhood(
    center_pos: Tuple[int, int, int],
    gt_canon: Dict[str, Any],
    agent_pos: Tuple[int, int, int],
    token_builder: TokenBuilder
) -> torch.Tensor:
    """
    Extract 3×3×3 neighborhood around center_pos and convert to feature tensor.
    
    Args:
        center_pos: (x, y, z) absolute position of center voxel
        gt_canon: Canonicalized ground truth grid dict
        agent_pos: (x, y, z) agent position
        token_builder: TokenBuilder instance
        
    Returns:
        Feature tensor [13, 3, 3, 3] where channels are:
        [occupancy, material_class, hazard_class, breakability, tool_hint,
         dx, dy, dz, age_s, support, passable, head_obstacle, thin_layer]
        
    Missing neighbors are zero-padded.
    """
    # Build position lookup
    pos_to_cell = {tuple(cell["pos"]): cell for cell in gt_canon["cells"]}
    
    # Fixed order: iterate y (low to high), then x, then z
    features = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                neighbor_pos = (center_pos[0] + dx, center_pos[1] + dy, center_pos[2] + dz)
                cell = pos_to_cell.get(neighbor_pos)
                
                if cell:
                    token = token_builder.from_canonical_cell(cell, agent_pos)
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


def process_training_record(record_path: str, margin: int = 2) -> List[Dict[str, Any]]:
    """
    Process a single training record into model samples.
    
    Args:
        record_path: Path to JSON training record
        margin: Voxel margin around partial grid bounding box (default: 2)
        
    Returns:
        List of samples (one per canonical cell)
    """
    record_start_time = time.time()
    logger.info(f"Processing record: {record_path}")
    
    # Load JSON
    load_start = time.time()
    with open(record_path) as f:
        record = json.load(f)
    load_time = time.time() - load_start
    logger.debug(f"  JSON load time: {load_time:.3f}s")
    
    canon = Canonicalize()
    token_builder = TokenBuilder()
    
    # Canonicalize partial grid (for reference, but we use GT for training)
    partial_start = time.time()
    partial_canon = canon.canonicalize(record["partial_grid"])
    partial_time = time.time() - partial_start
    center = partial_canon["center"]
    agent_pos = center  # Agent is at center
    num_partial_cells = len(partial_canon["cells"])
    logger.debug(f"  Partial grid canonicalization: {partial_time:.3f}s ({num_partial_cells} cells)")
    
    # Canonicalize ground truth grid
    if not record.get("ground_truth_grid") or not record["ground_truth_grid"].get("ok"):
        logger.warning(f"No valid ground truth in {record_path}, skipping")
        return []
    
    gt_start = time.time()
    gt_canon = canon.canonicalize_ground_truth(record["ground_truth_grid"])
    gt_time = time.time() - gt_start
    gt_grid = record["ground_truth_grid"]
    num_gt_cells_total = len(gt_canon["cells"])
    logger.info(f"  Ground truth canonicalization: {gt_time:.3f}s ({num_gt_cells_total} cells)")
    
    # Compute bounding box of partial cells with margin
    bbox_start = time.time()
    if not partial_canon["cells"]:
        logger.warning(f"No partial cells found, skipping")
        return []
    
    partial_positions = [cell["pos"] for cell in partial_canon["cells"]]
    min_x = min(p[0] for p in partial_positions) - margin
    max_x = max(p[0] for p in partial_positions) + margin
    min_y = min(p[1] for p in partial_positions) - margin
    max_y = max(p[1] for p in partial_positions) + margin
    min_z = min(p[2] for p in partial_positions) - margin
    max_z = max(p[2] for p in partial_positions) + margin
    
    bbox_time = time.time() - bbox_start
    logger.info(f"  Bounding box: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}], z=[{min_z}, {max_z}] (margin={margin})")
    logger.debug(f"  Bounding box computation: {bbox_time:.3f}s")
    
    # Filter GT cells to only those within bounding box
    filter_start = time.time()
    filtered_gt_cells = [
        cell for cell in gt_canon["cells"]
        if min_x <= cell["pos"][0] <= max_x and
           min_y <= cell["pos"][1] <= max_y and
           min_z <= cell["pos"][2] <= max_z
    ]
    filter_time = time.time() - filter_start
    num_gt_cells = len(filtered_gt_cells)
    reduction_factor = num_gt_cells_total / num_gt_cells if num_gt_cells > 0 else float('inf')
    logger.info(f"  Filtered GT cells: {num_gt_cells}/{num_gt_cells_total} ({filter_time:.3f}s, {reduction_factor:.1f}x reduction)")
    
    # Process filtered canonical cells (from ground truth for training)
    samples = []
    token_time_total = 0.0
    label_time_total = 0.0
    tensor_time_total = 0.0
    
    logger.info(f"  Processing {num_gt_cells} filtered canonical cells...")
    cell_start_time = time.time()
    
    for idx, cell in enumerate(filtered_gt_cells):
        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - cell_start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"    Processed {idx + 1}/{num_gt_cells} cells ({rate:.1f} cells/s, {elapsed:.1f}s elapsed)")
        
        # Build 3×3×3 neighborhood feature tensor
        tensor_start = time.time()
        center_pos = cell["pos"]
        neighborhood = get_3x3x3_neighborhood(center_pos, gt_canon, agent_pos, token_builder)
        batch = {"neighborhood": neighborhood}  # [13, 3, 3, 3]
        tensor_time_total += time.time() - tensor_start
        
        # Get center token for relative position
        token_start = time.time()
        center_token = token_builder.from_canonical_cell(cell, agent_pos)
        token_time_total += time.time() - token_start
        
        # Get ground truth labels for center voxel
        label_start = time.time()
        rel_x, rel_y, rel_z = center_token.dx, center_token.dy, center_token.dz
        labels = get_ground_truth_labels(record, rel_x, rel_y, rel_z)
        label_time_total += time.time() - label_start
        
        # StructureType targets (soft scores, one per StructureType)
        structure_type_scores = torch.tensor(
            [float(labels["structure_type_scores"].get(st.value, 0.0)) for st in STRUCTURE_TYPES],
            dtype=torch.float32
        )
        
        # Affordance targets (multi-label)
        aff_set = set(labels.get("affordances", []))
        affordance_multi = torch.tensor(
            [1.0 if st.value in aff_set else 0.0 for st in AFFORDANCE_TYPES],
            dtype=torch.float32
        )
        
        # Risks (multi-label + per-type severity)
        risk_present = torch.zeros(len(RISK_TYPES), dtype=torch.float32)
        risk_severity_by_type = torch.full((len(RISK_TYPES),), -100, dtype=torch.long)  # ignore by default
        
        # Reduce GT list -> per-type presence and max severity for that type
        risk_entries = labels.get("risks", [])
        for r in risk_entries:
            try:
                rtype = RiskType(r["type"])
                sev = Severity(r["severity"])
            except (ValueError, KeyError):
                continue
            if rtype not in RISK_TYPES:
                continue
            t_idx = RISK_TYPES.index(rtype)
            risk_present[t_idx] = 1.0
            sev_idx = SEVERITY_TO_INDEX.get(sev)
            if sev_idx is None:
                continue
            current = int(risk_severity_by_type[t_idx].item())
            if current == -100 or sev_idx > current:
                risk_severity_by_type[t_idx] = sev_idx
        
        targets = {
            "structure_type_scores": structure_type_scores,  # [len(StructureType)]
            "affordance_multi": affordance_multi,            # [len(AffordanceType)]
            "risk_present": risk_present,                    # [len(RiskType)]
            "risk_severity_by_type": risk_severity_by_type,  # [len(RiskType)] (ignore=-100 if absent)
            "valid_mask": torch.tensor(True, dtype=torch.bool),
        }
        
        sample = {"batch": batch, "targets": targets}
        # Validate before appending
        if "batch" not in sample or "targets" not in sample:
            raise ValueError(f"Created invalid sample: keys={list(sample.keys())}")
        samples.append(sample)
    
    cell_total_time = time.time() - cell_start_time
    record_total_time = time.time() - record_start_time
    
    logger.info(f"  Cell processing complete:")
    logger.info(f"    Total cells processed: {len(samples)}")
    logger.info(f"    Total time: {cell_total_time:.3f}s ({cell_total_time/len(samples)*1000:.2f}ms per cell)")
    logger.info(f"    Token conversion: {token_time_total:.3f}s ({token_time_total/len(samples)*1000:.2f}ms per cell)")
    logger.info(f"    Label generation: {label_time_total:.3f}s ({label_time_total/len(samples)*1000:.2f}ms per cell)")
    logger.info(f"    Tensor creation: {tensor_time_total:.3f}s ({tensor_time_total/len(samples)*1000:.2f}ms per cell)")
    logger.info(f"  Record processing total: {record_total_time:.3f}s")
    logger.info(f"Generated {len(samples)} samples from {record_path}")
    return samples


# -------------------------------------------------
# Entry
# -------------------------------------------------

def train(
    samples: List[Dict[str, Any]],
    output_dir: str = "./perception_ckpt",
    per_device_batch_size: int = 1,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    logging_steps: int = 50,
    save_steps: int = 500,
    fp16: bool = True,
) -> None:
    """
    Train the perception model.
    
    Args:
        samples: List of training samples
        output_dir: Directory to save checkpoints
        per_device_batch_size: Batch size per device
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        logging_steps: Steps between logging
        save_steps: Steps between checkpoints
        fp16: Use mixed precision training
    """
    if not samples:
        raise ValueError("No training samples provided")

    # If no W&B API key is configured, default to offline mode to avoid interactive login.
    # This keeps logging enabled but writes runs locally under ./wandb/.
    import os
    if not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "offline")
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    else:
        logger.warning("CUDA not available, using CPU (training will be slow)")
    
    logger.info(f"Initializing model with {len(samples)} training samples")
    
    # Validate all samples before creating dataset
    for i, sample in enumerate(samples[:10]):  # Check first 10 samples
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {i} is not a dict: {type(sample)}")
        if "batch" not in sample:
            raise ValueError(f"Sample {i} missing 'batch' key. Keys: {list(sample.keys())}")
        if "targets" not in sample:
            raise ValueError(f"Sample {i} missing 'targets' key. Keys: {list(sample.keys())}")
    
    # Check a random sample from the middle
    if len(samples) > 100:
        mid_idx = len(samples) // 2
        sample = samples[mid_idx]
        if "targets" not in sample:
            raise ValueError(f"Mid sample {mid_idx} missing 'targets' key. Keys: {list(sample.keys())}")
    
    model = PerceptionModel3DCNN()
    
    # Move model to device
    model = model.to(device)
    
    # pyright: ignore[reportGeneralTypeIssues]
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_steps=save_steps,
        fp16=fp16 and device == "cuda",  # Only use fp16 on CUDA
        report_to="wandb",
        logging_dir=f"{output_dir}/logs",
        save_total_limit=3,
        load_best_model_at_end=False,
        dataloader_pin_memory=device == "cuda",  # Pin memory for GPU
        remove_unused_columns=False,  # Don't strip keys we need (batch, targets)
    )
    
    trainer = PerceptionTrainer(
        model=model,
        args=args,
        train_dataset=PerceptionDataset(samples),
        data_collator=collate_fn,
    )
    
    logger.info("Starting training...")
    logger.info(f"Model device: {next(model.parameters()).device}")
    logger.info(f"Training arguments: batch_size={per_device_batch_size}, epochs={num_epochs}, lr={learning_rate}")
    logger.info(f"FP16 enabled: {fp16 and device == 'cuda'}")
    
    try:
        trainer.train()
        logger.info("Training completed successfully")
        
        # Save final model
        model_save_dir = Path("models/voxel_l0_model")
        model_save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving final model to {model_save_dir}")
        trainer.save_model(str(model_save_dir))
        logger.info(f"Model saved successfully to {model_save_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


def main():
    """
    Main entry point for training.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PerceptionModel")
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing training record JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./perception_ckpt",
        help="Output directory for checkpoints (default: ./perception_ckpt)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per device (default: 1)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Steps between logging (default: 50)"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Steps between checkpoints (default: 500)"
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable mixed precision training"
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise ValueError(f"Not a directory: {data_dir}")
    
    # Find all JSON files
    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {data_dir}")
    
    logger.info(f"Found {len(json_files)} training records in {data_dir}")
    
    # Process all records
    data_processing_start = time.time()
    all_samples = []
    for idx, json_file in enumerate(json_files):
        try:
            record_start = time.time()
            samples = process_training_record(str(json_file))
            record_time = time.time() - record_start
            
            # Validate sample structure before adding
            for i, sample in enumerate(samples):
                if not isinstance(sample, dict):
                    raise ValueError(f"Sample {i} from {json_file} is not a dict: {type(sample)}")
                if "batch" not in sample:
                    raise ValueError(f"Sample {i} from {json_file} missing 'batch' key. Keys: {list(sample.keys())}")
                if "targets" not in sample:
                    raise ValueError(f"Sample {i} from {json_file} missing 'targets' key. Keys: {list(sample.keys())}")
            
            all_samples.extend(samples)
            logger.info(f"Record {idx+1}/{len(json_files)}: {len(samples)} samples in {record_time:.2f}s")
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}", exc_info=True)
            continue
    
    data_processing_time = time.time() - data_processing_start
    
    if not all_samples:
        raise ValueError("No samples generated from training records")
    
    logger.info(f"=" * 60)
    logger.info(f"Data processing summary:")
    logger.info(f"  Total records processed: {len(json_files)}")
    logger.info(f"  Total samples generated: {len(all_samples)}")
    logger.info(f"  Total processing time: {data_processing_time:.2f}s")
    logger.info(f"  Average time per record: {data_processing_time/len(json_files):.2f}s")
    logger.info(f"  Average samples per record: {len(all_samples)/len(json_files):.0f}")
    logger.info(f"=" * 60)
    
    # Train
    train(
        samples=all_samples,
        output_dir=args.output_dir,
        per_device_batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=not args.no_fp16,
    )


if __name__ == "__main__":
    main()
