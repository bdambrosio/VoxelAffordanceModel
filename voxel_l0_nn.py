import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptionModel(nn.Module):
    """
    CanonicalGrid → per-voxel perceptual evidence fields.

    IMPORTANT:
    - No mutex structure labels.
    - Structure is inferred non-locally by the assembler.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        max_rel_dist: int = 4,

        # cue counts
        num_structure_cues: int = 5,     # surface, wall, enclosure, edge, continuity
        num_affordance_types: int = 10,  # incl NONE
        num_risk_types: int = 6,
        num_severity: int = 3,
    ):
        super().__init__()

        # -------------------------
        # Embeddings
        # -------------------------

        self.occupancy_emb = nn.Embedding(4, 8)
        self.material_emb = nn.Embedding(7, 8)
        self.hazard_emb = nn.Embedding(3, 4)
        self.break_emb = nn.Embedding(2, 2)
        self.tool_emb = nn.Embedding(5, 4)

        self.numeric_proj = nn.Linear(4, 32)  # dx, dy, dz, age

        self.input_proj = nn.Sequential(
            nn.Linear(8 + 8 + 4 + 2 + 4 + 32 + 4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # -------------------------
        # Transformer
        # -------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # -------------------------
        # Structure cue head (NON-MUTEX)
        # -------------------------
        # Independent sigmoid outputs
        self.structure_cue_head = nn.Linear(d_model, num_structure_cues)

        # -------------------------
        # Affordance proposal head
        # -------------------------
        self.affordance_head = nn.Linear(
            d_model,
            num_affordance_types + 4
            # type logits + confidence + local_target_offset(3)
        )

        # -------------------------
        # Risk proposal head
        # -------------------------
        self.risk_head = nn.Linear(
            d_model,
            num_risk_types + num_severity + 1
            # type logits + severity logits + confidence
        )

    def forward(self, batch):
        """
        batch fields (padded):
          occupancy, material_class, hazard_class, breakability, tool_hint
          numeric = [dx, dy, dz, age]
          geom = [support, passable, head_obstacle, thin_layer]
          valid_mask
        """

        emb = torch.cat([
            self.occupancy_emb(batch["occupancy"]),
            self.material_emb(batch["material_class"]),
            self.hazard_emb(batch["hazard_class"]),
            self.break_emb(batch["breakability"]),
            self.tool_emb(batch["tool_hint"]),
        ], dim=-1)

        num = self.numeric_proj(batch["numeric"])
        geom = batch["geom"].float()

        x = torch.cat([emb, num, geom], dim=-1)
        x = self.input_proj(x)

        key_padding_mask = ~batch["valid_mask"]

        x = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask
        )

        # -------------------------
        # Heads
        # -------------------------

        structure_cues = torch.sigmoid(
            self.structure_cue_head(x)
        )
        # shape: B × N × num_structure_cues

        affordance_out = self.affordance_head(x)
        risk_out = self.risk_head(x)

        return {
            "structure_cues": structure_cues,
            "affordance": affordance_out,
            "risk": risk_out,
        }
