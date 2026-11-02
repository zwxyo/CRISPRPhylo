from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from crispr_data_pipeline_zwx import (
    build_dataset_from_simulated_dir_zwx,
    collate_crispr_tree_samples_zwx,
    spacer_vector_dim_zwx,
)
from crispr_model_core_zwx import CRISPRRepeatAxialModelZwx, combined_supervised_loss_zwx
from tree_utils_zwx import normalized_rf_distance_zwx


def parse_args_zwx() -> argparse.Namespace:
    # Parse training script arguments.

    parser = argparse.ArgumentParser(description="Train CRISPR Axial Transformer with true tree supervision.")
    parser.add_argument("--data-dir", default="simulated_data_theta_1_8_seed42_array200")
    parser.add_argument("--output-dir", default="artifacts_zwx_earlystopping_1_8_array200")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kmer-size", type=int, default=4)
    parser.add_argument("--alignment-threshold", type=float, default=0.18)
    parser.add_argument("--max-msa-cols", type=int, default=256)
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--topology-weight", type=float, default=1.0)
    parser.add_argument("--max-train-groups", type=int, default=None)
    parser.add_argument("--max-valid-groups", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    # early stopping
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    return parser.parse_args()


def move_batch_to_device_zwx(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    # Move tensors in the batch to the specified device.

    moved: Dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def rank_vector_zwx(values: torch.Tensor) -> torch.Tensor:
    # Compute simple ranks for a one-dimensional tensor for Spearman correlation.

    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks


def spearman_corr_zwx(pred: torch.Tensor, target: torch.Tensor) -> float:
    # Compute the Spearman correlation coefficient for pairwise distances in a single sample.

    pred_rank = rank_vector_zwx(pred.detach().float())
    target_rank = rank_vector_zwx(target.detach().float())
    pred_rank = pred_rank - pred_rank.mean()
    target_rank = target_rank - target_rank.mean()
    denominator = pred_rank.norm() * target_rank.norm()
    if denominator.item() <= 1e-8:
        return 0.0
    return float((pred_rank * target_rank).sum() / denominator)


def upper_pair_values_zwx(matrix: torch.Tensor, valid_size: int) -> torch.Tensor:
    # Extract upper-triangular pairwise values within the valid array range.

    indices = torch.triu_indices(valid_size, valid_size, offset=1, device=matrix.device)
    return matrix[:valid_size, :valid_size][indices[0], indices[1]]


@torch.no_grad()
def evaluate_model_zwx(
    model: CRISPRRepeatAxialModelZwx,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    # Compute mean loss, Spearman correlation, and normalized RF distance on the validation set.

    model.eval()
    losses: List[float] = []
    distance_losses: List[float] = []
    topology_losses: List[float] = []
    spearman_values: List[float] = []
    rf_values: List[float] = []
    amp_context = torch.cuda.amp.autocast if device.type == "cuda" and use_amp else nullcontext
    for batch in loader:
        batch = move_batch_to_device_zwx(batch, device)
        with amp_context():
            outputs = model(batch)
            loss_dict = combined_supervised_loss_zwx(outputs, batch)
        losses.append(float(loss_dict["loss"].detach().cpu()))
        distance_losses.append(float(loss_dict["distance_loss"].detach().cpu()))
        topology_losses.append(float(loss_dict["topology_loss"].detach().cpu()))
        row_counts = batch["row_mask"].sum(dim=1).detach().cpu().tolist()
        for batch_index, valid_size in enumerate(row_counts):
            valid_size = int(valid_size)
            pred_values = upper_pair_values_zwx(outputs["pairwise_distance"][batch_index], valid_size)
            true_values = upper_pair_values_zwx(batch["target_distance"][batch_index], valid_size)
            spearman_values.append(spearman_corr_zwx(pred_values, true_values))
            rf_values.append(
                normalized_rf_distance_zwx(
                    batch["newick"][batch_index],
                    batch["strain_labels"][batch_index],
                    outputs["pairwise_distance"][batch_index, :valid_size, :valid_size],
                )
            )
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "distance_loss": sum(distance_losses) / max(1, len(distance_losses)),
        "topology_loss": sum(topology_losses) / max(1, len(topology_losses)),
        "spearman": sum(spearman_values) / max(1, len(spearman_values)),
        "rf": sum(rf_values) / max(1, len(rf_values)),
    }


def train_one_epoch_zwx(
    model: CRISPRRepeatAxialModelZwx,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    distance_weight: float,
    topology_weight: float,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = True,
) -> Dict[str, float]:
    # Train for one epoch and return the mean loss.

    model.train()
    total_loss = 0.0
    total_distance = 0.0
    total_topology = 0.0
    steps = 0
    amp_context = torch.cuda.amp.autocast if device.type == "cuda" and use_amp else nullcontext
    for batch in loader:
        batch = move_batch_to_device_zwx(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with amp_context():
            outputs = model(batch)
            loss_dict = combined_supervised_loss_zwx(
                outputs,
                batch,
                distance_weight=distance_weight,
                topology_weight=topology_weight,
            )
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss_dict["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += float(loss_dict["loss"].detach().cpu())
        total_distance += float(loss_dict["distance_loss"].detach().cpu())
        total_topology += float(loss_dict["topology_loss"].detach().cpu())
        steps += 1
    return {
        "loss": total_loss / max(1, steps),
        "distance_loss": total_distance / max(1, steps),
        "topology_loss": total_topology / max(1, steps),
    }


def build_loaders_zwx(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    # Build training and validation DataLoaders.

    train_dataset = build_dataset_from_simulated_dir_zwx(
        args.data_dir,
        split="train",
        kmer_size=args.kmer_size,
        alignment_threshold=args.alignment_threshold,
        max_groups=args.max_train_groups,
        max_msa_cols=args.max_msa_cols,
    )
    valid_dataset = build_dataset_from_simulated_dir_zwx(
        args.data_dir,
        split="valid",
        kmer_size=args.kmer_size,
        alignment_threshold=args.alignment_threshold,
        max_groups=args.max_valid_groups,
        max_msa_cols=args.max_msa_cols,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_crispr_tree_samples_zwx,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_crispr_tree_samples_zwx,
    )
    return train_loader, valid_loader


def main_zwx() -> None:
    # Run the full training pipeline and save the model with the best validation metric.

    args = parse_args_zwx()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, valid_loader = build_loaders_zwx(args)
    model = CRISPRRepeatAxialModelZwx(
        spacer_dim=spacer_vector_dim_zwx(args.kmer_size),
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    use_amp = device.type == "cuda" and not bool(args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"amp={use_amp} batch_size={args.batch_size} max_msa_cols={args.max_msa_cols}")
    best_score = float("-inf")
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch_zwx(
            model,
            train_loader,
            optimizer,
            device,
            distance_weight=args.distance_weight,
            topology_weight=args.topology_weight,
            scaler=scaler,
            use_amp=use_amp,
        )
        valid_metrics = evaluate_model_zwx(model, valid_loader, device, use_amp=use_amp)
        selection_score = valid_metrics["spearman"] - valid_metrics["rf"]
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"valid_spearman={valid_metrics['spearman']:.4f} "
            f"valid_rf={valid_metrics['rf']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f}"
        )
        torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, output_dir / "latest_zwx.pt")
        # if selection_score > best_score:
        #     best_score = selection_score
        #     torch.save(
        #         {"model_state_dict": model.state_dict(), "args": vars(args), "valid_metrics": valid_metrics},
        #         output_dir / "best_zwx.pt",
        #     )
        if selection_score > best_score + args.min_delta:
            best_score = selection_score
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "valid_metrics": valid_metrics,
                },
                output_dir / "best_zwx.pt",
            )

            print(f"new best score={best_score:.4f}")

        else:
            epochs_without_improvement += 1
            print(
                f"no improvement "
                f"({epochs_without_improvement}/{args.patience})"
            )

            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping triggered at epoch {epoch}, "
                    f"best_score={best_score:.4f}"
                )
                break
        
        # if valid_metrics["spearman"] >= 0.7 and valid_metrics["rf"] <= 0.3:
        #     print("validation target reached: spearman>=0.7 and rf<=0.3")


if __name__ == "__main__":
    main_zwx()
