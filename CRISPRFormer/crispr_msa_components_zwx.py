from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_position_encoding_zwx(length: int, dim: int, device: torch.device) -> torch.Tensor:
    # Generate sinusoidal positional encodings for MSA columns.

    positions = torch.arange(length, device=device, dtype=torch.float32)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    encoding = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
    encoding[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term)
    return encoding


class RepeatEncoderZwx(nn.Module):
    # Encode repeat sequences to provide context vectors for each repeat group.

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int = 2,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.token_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        self.input_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=model_dim,
                    nhead=num_heads,
                    dim_feedforward=model_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, repeat_tokens: torch.Tensor, repeat_mask: torch.Tensor) -> torch.Tensor:
        # Return the repeat embedding for each sample.

        repeat_len = repeat_tokens.shape[1]
        x = self.token_embedding(repeat_tokens)
        x = x + sinusoidal_position_encoding_zwx(repeat_len, self.model_dim, repeat_tokens.device).unsqueeze(0)
        x = self.dropout(self.input_norm(x))
        x = x * repeat_mask.unsqueeze(-1).to(x.dtype)
        padding_mask = ~repeat_mask
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)
            x = x * repeat_mask.unsqueeze(-1).to(x.dtype)
        token_counts = repeat_mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
        pooled = (x * repeat_mask.unsqueeze(-1).to(x.dtype)).sum(dim=1) / token_counts
        return self.output_norm(pooled)


class SelfAttentionZwx(nn.Module):
    # Wrap multi-head self-attention with batch_first layout.

    def __init__(self, model_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # Compute self-attention output using the specified padding mask.

        output, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return output


class AxialAttentionBlockZwx(nn.Module):
    # Alternate axial attention along the array and spacer-column dimensions.

    def __init__(self, model_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.row_norm = nn.LayerNorm(model_dim)
        self.row_attention = SelfAttentionZwx(model_dim, num_heads, dropout)
        self.col_norm = nn.LayerNorm(model_dim)
        self.col_attention = SelfAttentionZwx(model_dim, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, row_mask: torch.Tensor, col_mask: torch.Tensor) -> torch.Tensor:
        # Apply one layer of row attention, column attention, and FFN to the MSA tensor.

        batch_size, rows, cols, dim = x.shape
        row_input = self.row_norm(x).permute(0, 2, 1, 3).reshape(batch_size * cols, rows, dim)
        row_padding_mask = (~row_mask).unsqueeze(1).expand(batch_size, cols, rows).reshape(batch_size * cols, rows)
        row_output = self.row_attention(row_input, key_padding_mask=row_padding_mask)
        row_output = row_output.reshape(batch_size, cols, rows, dim).permute(0, 2, 1, 3)
        x = x + row_output

        col_input = self.col_norm(x).reshape(batch_size * rows, cols, dim)
        col_padding_mask = (~col_mask).unsqueeze(1).expand(batch_size, rows, cols).reshape(batch_size * rows, cols)
        col_output = self.col_attention(col_input, key_padding_mask=col_padding_mask)
        col_output = col_output.reshape(batch_size, rows, cols, dim)
        x = x + col_output
        return x + self.ffn(self.ffn_norm(x))


class ConservationAttentionPoolingZwx(nn.Module):
    # Pool spacer columns into array embeddings using conservation as an attention prior.

    def __init__(self, model_dim: int):
        super().__init__()
        self.score = nn.Linear(model_dim, 1)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        msa_mask: torch.Tensor,
        conservation_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Compute weighted pooling based on token content and column conservation.

        logits = self.score(token_embeddings).squeeze(-1)
        conservation_bias = torch.log(conservation_weights.clamp_min(1e-4)).unsqueeze(1)
        logits = logits + conservation_bias
        logits = logits.masked_fill(~msa_mask, -1e9)
        weights = torch.softmax(logits, dim=-1) * msa_mask.to(token_embeddings.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("brc,brcd->brd", weights, token_embeddings)


class RowReadoutZwx(nn.Module):
    # Read out each CRISPR array row as an array embedding.

    def __init__(self, model_dim: int):
        super().__init__()
        self.attention_pooling = ConservationAttentionPoolingZwx(model_dim)
        self.repeat_gate = nn.Linear(model_dim, model_dim)
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        msa_mask: torch.Tensor,
        repeat_embedding: torch.Tensor,
        row_mask: torch.Tensor,
        conservation_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Combine conserved-spacer attention and repeat gating to generate array embeddings.

        row_embeddings = self.attention_pooling(token_embeddings, msa_mask, conservation_weights)
        gate = torch.sigmoid(self.repeat_gate(repeat_embedding)).unsqueeze(1)
        row_embeddings = self.output_norm(row_embeddings * gate)
        return row_embeddings * row_mask.unsqueeze(-1).to(row_embeddings.dtype)


class PairwiseDistanceHeadZwx(nn.Module):
    # Predict tree distances between arrays in the same group from their embeddings.

    def __init__(self, model_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self,
        row_embeddings: torch.Tensor,
        repeat_embedding: torch.Tensor,
        row_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Return a symmetric, nonnegative pairwise distance matrix.

        left = row_embeddings.unsqueeze(2).expand(-1, -1, row_embeddings.size(1), -1)
        right = row_embeddings.unsqueeze(1).expand(-1, row_embeddings.size(1), -1, -1)
        repeat_context = repeat_embedding.unsqueeze(1).unsqueeze(1).expand(-1, row_embeddings.size(1), row_embeddings.size(1), -1)
        features = torch.cat([torch.abs(left - right), left * right, repeat_context], dim=-1)
        distances = F.softplus(self.mlp(features).squeeze(-1))
        distances = 0.5 * (distances + distances.transpose(1, 2))
        diagonal = torch.arange(distances.size(1), device=distances.device)
        distances[:, diagonal, diagonal] = 0.0
        pair_mask = row_mask.unsqueeze(2) & row_mask.unsqueeze(1)
        return distances.masked_fill(~pair_mask, 0.0)


__all__ = [
    "AxialAttentionBlockZwx",
    "ConservationAttentionPoolingZwx",
    "PairwiseDistanceHeadZwx",
    "RepeatEncoderZwx",
    "RowReadoutZwx",
    "SelfAttentionZwx",
    "sinusoidal_position_encoding_zwx",
]
