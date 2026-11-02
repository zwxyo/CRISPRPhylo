from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from crispr_msa_components_zwx import (
    AxialAttentionBlockZwx,
    PairwiseDistanceHeadZwx,
    RepeatEncoderZwx,
    RowReadoutZwx,
    sinusoidal_position_encoding_zwx,
)


class CRISPRRepeatAxialModelZwx(nn.Module):
    # Build an Axial Transformer from spacer MSA input and produce array embeddings.

    def __init__(
        self,
        spacer_dim: int,
        repeat_vocab_size: int = 6,
        model_dim: int = 192,
        num_heads: int = 6,
        num_layers: int = 4,
        dropout: float = 0.1,
        topology_temperature: float = 0.2,
    ):
        super().__init__()
        self.model_dim = int(model_dim)
        self.topology_temperature = float(topology_temperature)
        self.spacer_projection = nn.Sequential(
            nn.Linear(spacer_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.input_norm = nn.LayerNorm(model_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.column_scale = nn.Parameter(torch.tensor(1.0))
        self.repeat_encoder = RepeatEncoderZwx(
            vocab_size=repeat_vocab_size,
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.axial_blocks = nn.ModuleList(
            [AxialAttentionBlockZwx(model_dim=model_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.row_readout = RowReadoutZwx(model_dim=model_dim)
        self.group_projection = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.distance_head = PairwiseDistanceHeadZwx(model_dim=model_dim, dropout=dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Run the forward pass and return distance predictions and the topology distribution.

        msa_vectors = batch["msa_vectors"]
        msa_mask = batch["msa_mask"]
        row_mask = batch["row_mask"]
        col_mask = batch["col_mask"]
        conservation_weights = batch["conservation_weights"]
        repeat_tokens = batch["repeat_tokens"]
        repeat_mask = batch["repeat_mask"]
        cols = msa_vectors.shape[2]

        x = self.spacer_projection(msa_vectors)
        col_pos = sinusoidal_position_encoding_zwx(cols, self.model_dim, msa_vectors.device)
        x = x + self.column_scale * col_pos.view(1, 1, cols, self.model_dim)
        x = self.input_dropout(self.input_norm(x))
        x = x * msa_mask.unsqueeze(-1).to(x.dtype)

        repeat_embedding = self.repeat_encoder(repeat_tokens=repeat_tokens, repeat_mask=repeat_mask)
        for block in self.axial_blocks:
            x = block(x, row_mask=row_mask, col_mask=col_mask)
            x = x * msa_mask.unsqueeze(-1).to(x.dtype)

        row_embeddings = self.row_readout(
            token_embeddings=x,
            msa_mask=msa_mask,
            repeat_embedding=repeat_embedding,
            row_mask=row_mask,
            conservation_weights=conservation_weights,
        )
        valid_rows = row_mask.sum(dim=1, keepdim=True).clamp_min(1).to(row_embeddings.dtype)
        group_embedding = self.group_projection(row_embeddings.sum(dim=1) / valid_rows)
        pairwise_distance = self.distance_head(row_embeddings, repeat_embedding, row_mask)
        topology_log_probs = pairwise_topology_log_probs_zwx(
            pairwise_distance=pairwise_distance,
            row_mask=row_mask,
            temperature=self.topology_temperature,
        )
        return {
            "row_embeddings": row_embeddings,
            "array_embedding": row_embeddings,
            "group_embedding": group_embedding,
            "repeat_embedding": repeat_embedding,
            "pairwise_distance": pairwise_distance,
            "pairwise_distance_logits": pairwise_distance,
            "topology_log_probs": topology_log_probs,
        }


def _ordered_pair_mask_zwx(row_mask: torch.Tensor) -> torch.Tensor:
    # Build a mask for valid ordered array pairs, excluding the diagonal.

    pair_mask = row_mask.unsqueeze(2) & row_mask.unsqueeze(1)
    diagonal = torch.eye(row_mask.size(1), dtype=torch.bool, device=row_mask.device).unsqueeze(0)
    return pair_mask & ~diagonal


def pairwise_topology_log_probs_zwx(
    pairwise_distance: torch.Tensor,
    row_mask: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    # Convert predicted distances into a log-probability matrix for the topology KL loss.

    pair_mask = _ordered_pair_mask_zwx(row_mask)
    logits = -pairwise_distance / max(float(temperature), 1e-6)
    logits = logits.masked_fill(~pair_mask, -1e9)
    flat_log_probs = torch.log_softmax(logits.flatten(1), dim=1)
    return flat_log_probs.view_as(pairwise_distance).masked_fill(~pair_mask, 0.0)


def masked_distance_mse_loss_zwx(
    pred_distance: torch.Tensor,
    true_distance: torch.Tensor,
    row_mask: torch.Tensor,
) -> torch.Tensor:
    # Compute the MSE loss using ground-truth tree distances as supervision.

    pair_mask = _ordered_pair_mask_zwx(row_mask)
    return F.mse_loss(pred_distance[pair_mask], true_distance[pair_mask])


def topology_kl_loss_zwx(
    topology_log_probs: torch.Tensor,
    target_topology: torch.Tensor,
    row_mask: torch.Tensor,
) -> torch.Tensor:
    # Compute the KL divergence loss using the ground-truth tree topology distribution as supervision.

    pair_mask = _ordered_pair_mask_zwx(row_mask)
    pred = topology_log_probs.flatten(1)
    target = target_topology.masked_fill(~pair_mask, 0.0)
    target = target / target.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-8).view(-1, 1, 1)
    return F.kl_div(pred, target.flatten(1), reduction="batchmean")


def combined_supervised_loss_zwx(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    distance_weight: float = 1.0,
    topology_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    # Combine the distance MSE and topology KL objectives.

    distance_loss = masked_distance_mse_loss_zwx(
        outputs["pairwise_distance"],
        batch["target_distance"],
        batch["row_mask"],
    )
    topology_loss = topology_kl_loss_zwx(
        outputs["topology_log_probs"],
        batch["target_topology"],
        batch["row_mask"],
    )
    loss = float(distance_weight) * distance_loss + float(topology_weight) * topology_loss
    return {
        "loss": loss,
        "distance_loss": distance_loss,
        "topology_loss": topology_loss,
    }


def build_tree_aware_contrastive_loss_zwx(
    row_embeddings: torch.Tensor,
    target_pairwise_dist: torch.Tensor,
    row_mask: torch.Tensor,
    temperature: float = 0.2,
    positive_k: int = 2,
) -> torch.Tensor:
    # Tree-distance-aware contrastive loss.

    normalized_embeddings = F.normalize(row_embeddings, p=2, dim=-1)
    batch_losses: list[torch.Tensor] = []
    for batch_index in range(row_embeddings.size(0)):
        valid_indices = torch.nonzero(row_mask[batch_index], as_tuple=False).squeeze(-1)
        embeddings = normalized_embeddings[batch_index, valid_indices]
        distances = target_pairwise_dist[batch_index][valid_indices][:, valid_indices]
        similarities = embeddings @ embeddings.transpose(0, 1)
        sample_losses: list[torch.Tensor] = []
        for anchor_index in range(valid_indices.numel()):
            candidate_distances = distances[anchor_index].clone()
            candidate_distances[anchor_index] = torch.inf
            positive_count = min(int(positive_k), max(0, valid_indices.numel() - 1))
            for positive_index in torch.argsort(candidate_distances)[:positive_count]:
                logits = similarities[anchor_index] / temperature
                keep_mask = torch.ones_like(logits, dtype=torch.bool)
                keep_mask[anchor_index] = False
                kept_logits = logits[keep_mask]
                kept_indices = torch.arange(logits.size(0), device=logits.device)[keep_mask]
                positive_position = torch.nonzero(kept_indices == positive_index, as_tuple=False).squeeze(-1)
                sample_losses.append(-(kept_logits[positive_position] - torch.logsumexp(kept_logits, dim=0)).mean())
        if sample_losses:
            batch_losses.append(torch.stack(sample_losses).mean())
    if not batch_losses:
        return row_embeddings.sum() * 0.0
    return torch.stack(batch_losses).mean()


__all__ = [
    "CRISPRRepeatAxialModelZwx",
    "build_tree_aware_contrastive_loss_zwx",
    "combined_supervised_loss_zwx",
    "masked_distance_mse_loss_zwx",
    "pairwise_topology_log_probs_zwx",
    "topology_kl_loss_zwx",
]
