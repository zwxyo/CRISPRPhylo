from __future__ import annotations
from dataclasses import dataclass

from graph_msa_dataset import ArrayPath, attach_spacer_class_ids
from sklearn.cluster import DBSCAN

@dataclass(frozen=True)
class SpacerClass:
    spacer_class_id: str
    repeat_family: str
    canonical_sequence: str
    member_sequences: list[str]
    support_count: int


@dataclass
class RepeatFamilyGraph:
    repeat_family: str
    nodes: dict[str, dict]
    edges: dict[tuple[str, str], dict]
    array_paths: dict[str, list[str]]

# Compute the Hamming distance.
# def hamming_distance(left: str, right: str) -> int:
#     return sum(1 for left_base, right_base in zip(left, right) if left_base != right_base)


# def sequence_similarity(left: str, right: str) -> float:
#     if len(left) != len(right):
#         return 0.0
#     if not left:
#         return 1.0
#     return 1.0 - hamming_distance(left, right) / len(left)



def levenshtein_distance(seq1, seq2):
    len1 = len(seq1) + 1
    len2 = len(seq2) + 1

    # Create a matrix
    # matrix = np.zeros((len1, len2), dtype=int)
    matrix = [[0] * len2 for _ in range(len1)]

    # Initialize the matrix
    for i in range(len1):
        matrix[i][0] = i
    for j in range(len2):
        matrix[0][j] = j

    # Compute the Levenshtein distance
    for i in range(1, len1):
        for j in range(1, len2):
            if seq1[i - 1] == seq2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,  # Deletion
                               matrix[i][j - 1] + 1,  # Insertion
                               matrix[i - 1][j - 1] + cost)  # Substitution

    return matrix[-1][-1]

def sequence_similarity(left: str, right: str, max_levenshtein_distance: int) -> float:

    if len(left) == 0:
        return 1.0
     
    dist = levenshtein_distance(left, right)

    return 1.0 if dist <= max_levenshtein_distance else 0.0


# Group spacer sequences into spacer classes.
# Prioritize frequent spacers as canonical sequences, then assign less frequent sequences to existing canonical sequences where possible.
def build_spacer_classes(
    array_paths: list[ArrayPath],
    max_levenshtein_distance: int,
) -> tuple[list[ArrayPath], dict[str, SpacerClass]]:
    repeat_family = array_paths[0].repeat_family
    sequence_counts: dict[str, int] = {}
    for array_path in array_paths:
        for sequence in array_path.spacer_sequences:
            sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1

    ordered_sequences = sorted(sequence_counts, key=lambda sequence: (-sequence_counts[sequence], sequence))
    sequence_to_class_id: dict[str, str] = {}
    spacer_classes: dict[str, SpacerClass] = {}
    class_canonical_sequences: dict[str, str] = {}
    class_member_sequences: dict[str, list[str]] = {}

    for sequence in ordered_sequences:
        assigned_class_id = ""
        for spacer_class_id, canonical_sequence in class_canonical_sequences.items():
            if len(sequence) != len(canonical_sequence):
                continue
            if levenshtein_distance(sequence, canonical_sequence) <= max_levenshtein_distance:
                assigned_class_id = spacer_class_id
                break
        if not assigned_class_id:
            assigned_class_id = f"{repeat_family}_spacer_{len(class_canonical_sequences) + 1:05d}"
            class_canonical_sequences[assigned_class_id] = sequence
            class_member_sequences[assigned_class_id] = []
        class_member_sequences[assigned_class_id].append(sequence)
        sequence_to_class_id[sequence] = assigned_class_id

    annotated_paths = attach_spacer_class_ids(array_paths, sequence_to_class_id)

    class_support_arrays: dict[str, set[str]] = {class_id: set() for class_id in class_canonical_sequences}
    for array_path in annotated_paths:
        for class_id in set(array_path.spacer_class_ids):
            class_support_arrays[class_id].add(array_path.array_uid)

    for spacer_class_id, canonical_sequence in class_canonical_sequences.items():
        spacer_classes[spacer_class_id] = SpacerClass(
            spacer_class_id=spacer_class_id,
            repeat_family=repeat_family,
            canonical_sequence=canonical_sequence,
            member_sequences=sorted(class_member_sequences[spacer_class_id]),
            support_count=len(class_support_arrays[spacer_class_id]),
        )

    return annotated_paths, spacer_classes

# Build a graph for the entire repeat family.
def build_repeat_family_graph(
    array_paths: list[ArrayPath],
    spacer_classes: dict[str, SpacerClass],
) -> RepeatFamilyGraph:
    repeat_family = array_paths[0].repeat_family
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], dict] = {}
    path_lookup: dict[str, list[str]] = {}

    for spacer_class_id, spacer_class in spacer_classes.items():
        nodes[spacer_class_id] = {
            "spacer_class_id": spacer_class_id,
            "repeat_family": spacer_class.repeat_family,
            "canonical_sequence": spacer_class.canonical_sequence,
            "member_sequences": list(spacer_class.member_sequences),
            "support_count": spacer_class.support_count,
            "array_uids": set(),
            "predecessors": set(),
            "successors": set(),
        }

    for array_path in array_paths:
        path_lookup[array_path.array_uid] = list(array_path.spacer_class_ids)
        for class_id in set(array_path.spacer_class_ids):
            nodes[class_id]["array_uids"].add(array_path.array_uid)
        for left_class_id, right_class_id in zip(array_path.spacer_class_ids[:-1], array_path.spacer_class_ids[1:]):
            nodes[left_class_id]["successors"].add(right_class_id)
            nodes[right_class_id]["predecessors"].add(left_class_id)
            edge_key = (left_class_id, right_class_id)
            if edge_key not in edges:
                edges[edge_key] = {
                    "source": left_class_id,
                    "target": right_class_id,
                    "support_count": 0,
                    "array_uids": set(),
                }
            edges[edge_key]["array_uids"].add(array_path.array_uid)

    for edge_payload in edges.values():
        edge_payload["support_count"] = len(edge_payload["array_uids"])

    return RepeatFamilyGraph(
        repeat_family=repeat_family,
        nodes=nodes,
        edges=edges,
        array_paths=path_lookup,
    )
