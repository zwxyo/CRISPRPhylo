from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graph_msa_dataset import ArrayPath
from spacer_graph import RepeatFamilyGraph, sequence_similarity

# Alignment weight parameters.
@dataclass(frozen=True)
class AlignmentWeights:
    match_reward: float = 2.0
    substitution_penalty: float = 1.0
    sequence_similarity_weight: float = 1.0
    context_weight: float = 0.5
    position_weight: float = 0.25
    gap_penalty: float = 1.0
    duplication_penalty: float = 0.75
    rearrangement_penalty: float = 1.5
    low_support_gap_discount: float = 0.25


@dataclass
class PairwiseAlignmentResult:
    left_array_uid: str
    right_array_uid: str
    matched_pairs: list[tuple[int, int]]
    left_gaps: list[int]
    right_gaps: list[int]
    alignment_score: float
    distance: float
    event_summary: dict[str, float]


def _normalized_position(index: int, length: int) -> float:
    if length == 1:
        return 0.0
    return index / (length - 1)


def _overlap_score(left_neighbors: set[str], right_neighbors: set[str]) -> float:
    if not left_neighbors and not right_neighbors:
        return 1.0
    union = left_neighbors | right_neighbors
    intersection = left_neighbors & right_neighbors
    return len(intersection) / len(union)


def _match_score(
    left_path: ArrayPath,
    right_path: ArrayPath,
    left_index: int,
    right_index: int,
    family_graph: RepeatFamilyGraph,
    weights: AlignmentWeights,
    max_levenshtein_distance: int,
) -> tuple[float, float]:
    left_class_id = left_path.spacer_class_ids[left_index]
    right_class_id = right_path.spacer_class_ids[right_index]
    left_node = family_graph.nodes[left_class_id]
    right_node = family_graph.nodes[right_class_id]

    left_sequence = left_path.spacer_sequences[left_index]
    right_sequence = right_path.spacer_sequences[right_index]
    seq_similarity = sequence_similarity(left_sequence, right_sequence, max_levenshtein_distance)

    predecessor_overlap = _overlap_score(left_node["predecessors"], right_node["predecessors"])
    successor_overlap = _overlap_score(left_node["successors"], right_node["successors"])
    context_similarity = 0.5 * (predecessor_overlap + successor_overlap)

    position_similarity = 1.0 - abs(
        _normalized_position(left_index, len(left_path.spacer_class_ids))
        - _normalized_position(right_index, len(right_path.spacer_class_ids))
    )

    score = 0.0
    if left_class_id == right_class_id:
        score += weights.match_reward
    else:
        score -= weights.substitution_penalty
    score += weights.sequence_similarity_weight * (2.0 * seq_similarity - 1.0)
    score += weights.context_weight * context_similarity
    score += weights.position_weight * position_similarity

    return score, seq_similarity


def _gap_cost(
    class_id: str,
    family_graph: RepeatFamilyGraph,
    weights: AlignmentWeights,
) -> float:
    support_count = float(family_graph.nodes[class_id]["support_count"])
    max_support_count = max(float(node["support_count"]) for node in family_graph.nodes.values())
    support_ratio = support_count / max_support_count
    return weights.gap_penalty * (1.0 - weights.low_support_gap_discount * (1.0 - support_ratio))

# Align two arrays using a Needleman-Wunsch-like algorithm with a more complex scoring scheme.
def align_array_paths(
    left_path: ArrayPath,
    right_path: ArrayPath,
    family_graph: RepeatFamilyGraph,
    weights: AlignmentWeights,
    max_levenshtein_distance: int,
) -> PairwiseAlignmentResult:
    left_length = len(left_path.spacer_class_ids)
    right_length = len(right_path.spacer_class_ids)

    dp = np.zeros((left_length + 1, right_length + 1), dtype=np.float64)
    trace = np.empty((left_length + 1, right_length + 1), dtype=object)
    trace[0, 0] = "START"

    for left_index in range(1, left_length + 1):
        class_id = left_path.spacer_class_ids[left_index - 1]
        dp[left_index, 0] = dp[left_index - 1, 0] - _gap_cost(class_id, family_graph, weights)
        trace[left_index, 0] = "UP"

    for right_index in range(1, right_length + 1):
        class_id = right_path.spacer_class_ids[right_index - 1]
        dp[0, right_index] = dp[0, right_index - 1] - _gap_cost(class_id, family_graph, weights)
        trace[0, right_index] = "LEFT"

    for left_index in range(1, left_length + 1):
        for right_index in range(1, right_length + 1):
            match_score, _ = _match_score(
                left_path,
                right_path,
                left_index - 1,
                right_index - 1,
                family_graph,
                weights,
                max_levenshtein_distance
            )
            diagonal_score = dp[left_index - 1, right_index - 1] + match_score
            up_score = dp[left_index - 1, right_index] - _gap_cost(
                left_path.spacer_class_ids[left_index - 1],
                family_graph,
                weights,
            )
            left_score = dp[left_index, right_index - 1] - _gap_cost(
                right_path.spacer_class_ids[right_index - 1],
                family_graph,
                weights,
            )

            best_score = diagonal_score
            best_trace = "DIAG"
            if up_score > best_score:
                best_score = up_score
                best_trace = "UP"
            if left_score > best_score:
                best_score = left_score
                best_trace = "LEFT"

            dp[left_index, right_index] = best_score
            trace[left_index, right_index] = best_trace

    matched_pairs: list[tuple[int, int]] = []
    left_gaps: list[int] = []
    right_gaps: list[int] = []
    match_count = 0
    substitution_count = 0
    sequence_divergences: list[float] = []

    left_index = left_length
    right_index = right_length
    while left_index > 0 or right_index > 0:
        step = trace[left_index, right_index]
        if step == "DIAG":
            current_left_index = left_index - 1
            current_right_index = right_index - 1
            matched_pairs.append((current_left_index, current_right_index))
            _, seq_similarity = _match_score(
                left_path,
                right_path,
                current_left_index,
                current_right_index,
                family_graph,
                weights,
                max_levenshtein_distance
            )
            if left_path.spacer_class_ids[current_left_index] == right_path.spacer_class_ids[current_right_index]:
                match_count += 1
            else:
                substitution_count += 1
            sequence_divergences.append(1.0 - seq_similarity)
            left_index -= 1
            right_index -= 1
            continue
        if step == "UP":
            left_gaps.append(left_index - 1)
            left_index -= 1
            continue
        if step == "LEFT":
            right_gaps.append(right_index - 1)
            right_index -= 1
            continue
        raise ValueError(f"Unexpected traceback step: {step}")

    matched_pairs.reverse()
    left_gaps.reverse()
    right_gaps.reverse()

    unmatched_rate = (len(left_gaps) + len(right_gaps)) / (left_length + right_length)
    substitution_rate = substitution_count / max(len(matched_pairs), 1)
    sequence_divergence = float(np.mean(sequence_divergences)) if sequence_divergences else 0.0
    event_penalty = (
        substitution_count
        + len(left_gaps)
        + len(right_gaps)
        + weights.duplication_penalty * 0.0
        + weights.rearrangement_penalty * 0.0
    ) / (left_length + right_length)
    distance = unmatched_rate + event_penalty + 0.5 * sequence_divergence

    return PairwiseAlignmentResult(
        left_array_uid=left_path.array_uid,
        right_array_uid=right_path.array_uid,
        matched_pairs=matched_pairs,
        left_gaps=left_gaps,
        right_gaps=right_gaps,
        alignment_score=float(dp[left_length, right_length]),
        distance=float(distance),
        event_summary={
            "match_count": float(match_count),
            "substitution_count": float(substitution_count),
            "left_gap_count": float(len(left_gaps)),
            "right_gap_count": float(len(right_gaps)),
            "unmatched_rate": float(unmatched_rate),
            "substitution_rate": float(substitution_rate),
            "sequence_divergence": float(sequence_divergence),
            "event_penalty": float(event_penalty),
        },
    )

# Generate a distance matrix within the family.
def build_pairwise_distance_matrix(
    array_paths: list[ArrayPath],
    family_graph: RepeatFamilyGraph,
    weights: AlignmentWeights,
    max_levenshtein_distance: int,
) -> tuple[list[str], np.ndarray, dict[tuple[str, str], PairwiseAlignmentResult]]:
    labels = [array_path.array_uid for array_path in array_paths]
    distance_matrix = np.zeros((len(array_paths), len(array_paths)), dtype=np.float64)
    alignment_results: dict[tuple[str, str], PairwiseAlignmentResult] = {}

    for left_index, left_path in enumerate(array_paths):
        for right_index in range(left_index + 1, len(array_paths)):
            right_path = array_paths[right_index]
            result = align_array_paths(left_path, right_path, family_graph, weights, max_levenshtein_distance)
            distance_matrix[left_index, right_index] = result.distance
            distance_matrix[right_index, left_index] = result.distance
            alignment_results[(left_path.array_uid, right_path.array_uid)] = result

    return labels, distance_matrix, alignment_results
