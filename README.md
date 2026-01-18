# Voxel Affordance Model

A lowest-level perceptual processor for Minecraft that transforms raw block-size voxel observations into structured perceptual representations suitable for navigation and decision-making.

## Overview

This system processes locally viewed, partially observable "bubbles" of block-size voxels accumulated by a [Cognitive_workbench](https://github.com/cognitive-workbench) Minecraft agent. It performs level 0 (L0) perceptual processing, converting raw voxel data into a relational representation of structures, affordances, and risks.

## Architecture

The perceptual pipeline consists of:

1. **Canonicalization** (`canonicalize.py`): Normalizes raw Minecraft block names and properties into a stable feature vocabulary
2. **Perception Model** (`voxel_l0_cnn.py`): A 3D CNN that processes 3×3×3 voxel neighborhoods to predict perceptual labels
3. **Perceptual Assembly** (`perceptual_frame.py`): Groups voxel-level predictions into structured `PerceptionFrame` objects containing:
   - **Structures**: Spatially grouped entities (surfaces, walls, slopes, cliffs, tunnels, rooms, voids)
   - **Affordances**: Available actions (step forward, jump up, drop down, dig, swim, surface, crouch, place)
   - **Risks**: Environmental hazards (fall, suffocation, drowning, lava)

## Key Features

- **Relational representation**: Output is variable-cardinality and relational, not grid-based
- **Deterministic assembly**: No learning in the assembly phase; purely rule-based grouping
- **Opaque IDs**: Structures use opaque identifiers for planner reference
- **No geometry/physics exposure**: Higher-level planners receive semantic abstractions only

## Usage

### Training

```bash
python training_harness.py training_data/ --output-dir ./perception_ckpt
```

### Inference

```bash
python voxel_l0_inference.py [path/to/sample.json]
# Or use first file from training_data automatically:
python voxel_l0_inference.py
```

## Data Format

Input: Raw partial grid observations from Cognitive_workbench agent:
```json
{
  "radius": 10,
  "center": {"x": 0, "y": 64, "z": 0},
  "cells": {
    "x,y,z": {
      "name": "minecraft:block[properties]",
      "ts": 1234567890.0
    }
  }
}
```

Output: `PerceptionFrame` with structures, affordances, and risks ready for planner consumption.

## Project Structure

- `perceptual_frame.py`: Core data structures and ontology
- `canonicalize.py`: Voxel feature normalization
- `voxel_l0_cnn.py`: 3D CNN perception model
- `training_harness.py`: Training pipeline
- `voxel_l0_inference.py`: Inference pipeline
- `ground_truth_labels.py`: Ground truth label generation
- `tokens.py`: Token representation for model input

## Requirements

See `requirements.txt` for Python dependencies.

## License

See `LICENSE` file for details.
