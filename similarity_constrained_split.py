"""Similarity-constrained, scaffold-disjoint molecular splitting."""

from __future__ import annotations

from collections import defaultdict
from math import floor
from typing import Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


class SimilarityConstrainedSplitError(RuntimeError):
    """Raised when the requested constrained split cannot be constructed."""


class _UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return int(item)

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _validate_inputs(
    mols: Sequence[Chem.Mol],
    labels: Sequence[int],
    split_ratios: tuple[float, float, float],
    max_mean_similarity: float,
    max_pair_similarity: float,
) -> None:
    if len(mols) != len(labels):
        raise ValueError("mols and labels must have the same length")
    if len(mols) < 3:
        raise ValueError("At least three molecules are required")
    if len(split_ratios) != 3 or not np.isclose(sum(split_ratios), 1.0):
        raise ValueError("split_ratios must contain three values summing to 1")
    if any(ratio <= 0 for ratio in split_ratios):
        raise ValueError("All split ratios must be positive")
    if not 0 < max_mean_similarity < 1:
        raise ValueError("max_mean_similarity must be between 0 and 1")
    if not 0 < max_pair_similarity < 1:
        raise ValueError("max_pair_similarity must be between 0 and 1")
    if any(mol is None for mol in mols):
        raise ValueError("All molecules must be valid RDKit molecules")


def _build_similarity_matrix(
    mols: Sequence[Chem.Mol], fingerprint_bits: int
) -> np.ndarray:
    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fingerprint_bits)
        for mol in mols
    ]
    similarities = np.empty((len(fingerprints), len(fingerprints)), dtype=np.float64)
    for index, fingerprint in enumerate(fingerprints):
        similarities[index] = DataStructs.BulkTanimotoSimilarity(
            fingerprint, fingerprints
        )
    return similarities


def _build_indivisible_components(
    mols: Sequence[Chem.Mol],
    similarities: np.ndarray,
    max_pair_similarity: float,
) -> tuple[list[list[int]], list[str]]:
    """Join equal scaffolds and every molecular pair at or above the max limit."""
    union_find = _UnionFind(len(mols))
    scaffold_first_index: dict[str, int] = {}
    scaffolds: list[str] = []

    for index, mol in enumerate(mols):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        scaffolds.append(scaffold)
        if scaffold in scaffold_first_index:
            union_find.union(scaffold_first_index[scaffold], index)
        else:
            scaffold_first_index[scaffold] = index

    for left in range(len(mols)):
        neighbors = np.flatnonzero(
            similarities[left, left + 1 :] >= max_pair_similarity
        )
        for offset in neighbors:
            union_find.union(left, left + 1 + int(offset))

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(mols)):
        grouped[union_find.find(index)].append(index)

    return list(grouped.values()), scaffolds


def _external_similarity_bounds(
    components: Sequence[Sequence[int]], similarities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative per-component sums/maxima against all other components."""
    all_indices = np.arange(similarities.shape[0])
    external_sums = np.empty(len(components), dtype=float)
    external_maxima = np.empty(len(components), dtype=float)

    for component_index, component in enumerate(components):
        component_indices = np.asarray(component, dtype=int)
        outside_indices = np.setdiff1d(
            all_indices, component_indices, assume_unique=True
        )
        if not len(outside_indices):
            raise SimilarityConstrainedSplitError(
                "All molecules form one indivisible similarity/scaffold component"
            )
        molecule_maxima = similarities[
            np.ix_(component_indices, outside_indices)
        ].max(axis=1)
        external_sums[component_index] = molecule_maxima.sum()
        external_maxima[component_index] = molecule_maxima.max()

    return external_sums, external_maxima


def _cross_split_metrics(
    source_indices: np.ndarray,
    reference_indices: np.ndarray,
    similarities: np.ndarray,
) -> dict[str, float]:
    maxima = similarities[np.ix_(source_indices, reference_indices)].max(axis=1)
    return {
        "mean_max": float(maxima.mean()),
        "median_max": float(np.median(maxima)),
        "p95_max": float(np.quantile(maxima, 0.95)),
        "maximum": float(maxima.max()),
    }


def make_similarity_constrained_scaffold_split(
    mols: Sequence[Chem.Mol],
    labels: Sequence[int],
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    max_mean_similarity: float = 0.5,
    max_pair_similarity: float = 0.8,
    seed: int = 0,
    fingerprint_bits: int = 2048,
    solver_time_limit: float = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Create an ECFP4-constrained, Murcko-scaffold-disjoint split.

    Molecular pairs with Tanimoto similarity greater than or equal to
    ``max_pair_similarity`` and molecules with the same Murcko scaffold are
    joined into indivisible components. A mixed-integer optimizer assigns
    components to validation and test while enforcing split sizes, class
    balance, and conservative upper bounds on held-out-to-training average
    maximum similarity.
    """
    _validate_inputs(
        mols,
        labels,
        split_ratios,
        max_mean_similarity,
        max_pair_similarity,
    )
    labels_array = np.asarray(labels, dtype=int)
    if not set(np.unique(labels_array)).issubset({0, 1}):
        raise ValueError("labels must be binary values 0 and 1")

    validation_size = floor(len(mols) * split_ratios[1])
    test_size = floor(len(mols) * split_ratios[2])
    train_size = len(mols) - validation_size - test_size
    global_positive_fraction = float(labels_array.mean())
    validation_positive_target = round(validation_size * global_positive_fraction)
    test_positive_target = round(test_size * global_positive_fraction)

    similarities = _build_similarity_matrix(mols, fingerprint_bits)
    components, scaffolds = _build_indivisible_components(
        mols, similarities, max_pair_similarity
    )
    component_sizes = np.asarray(
        [len(component) for component in components], dtype=float
    )
    component_positives = np.asarray(
        [labels_array[component].sum() for component in components], dtype=float
    )
    external_sums, external_maxima = _external_similarity_bounds(
        components, similarities
    )

    if external_maxima.max() >= max_pair_similarity:
        raise SimilarityConstrainedSplitError(
            "Internal component construction failed the maximum-similarity bound"
        )

    component_count = len(components)
    variable_count = 2 * component_count
    rng = np.random.default_rng(seed)
    tie_break = rng.uniform(0.0, 1e-9, size=variable_count)
    objective = np.concatenate([external_sums, external_sums]) + tie_break

    constraint_matrix = lil_matrix(
        (6 + component_count, variable_count), dtype=float
    )
    lower_bounds = np.full(6 + component_count, -np.inf)
    upper_bounds = np.full(6 + component_count, np.inf)

    constraint_matrix[0, :component_count] = component_sizes
    lower_bounds[0] = upper_bounds[0] = validation_size
    constraint_matrix[1, component_count:] = component_sizes
    lower_bounds[1] = upper_bounds[1] = test_size

    constraint_matrix[2, :component_count] = component_positives
    lower_bounds[2] = upper_bounds[2] = validation_positive_target
    constraint_matrix[3, component_count:] = component_positives
    lower_bounds[3] = upper_bounds[3] = test_positive_target

    constraint_matrix[4, :component_count] = external_sums
    upper_bounds[4] = np.nextafter(
        max_mean_similarity * validation_size, -np.inf
    )
    constraint_matrix[5, component_count:] = external_sums
    upper_bounds[5] = np.nextafter(max_mean_similarity * test_size, -np.inf)

    for component_index in range(component_count):
        row = 6 + component_index
        constraint_matrix[row, component_index] = 1
        constraint_matrix[row, component_count + component_index] = 1
        upper_bounds[row] = 1

    result = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=int),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(
            constraint_matrix.tocsr(), lower_bounds, upper_bounds
        ),
        options={"time_limit": solver_time_limit},
    )
    if not result.success:
        raise SimilarityConstrainedSplitError(
            "No constrained scaffold split satisfied the requested sizes, class "
            f"balance, and similarity limits: {result.message}"
        )

    validation_components = np.flatnonzero(result.x[:component_count] > 0.5)
    test_components = np.flatnonzero(result.x[component_count:] > 0.5)
    validation_indices = np.concatenate(
        [components[index] for index in validation_components]
    ).astype(int)
    test_indices = np.concatenate(
        [components[index] for index in test_components]
    ).astype(int)
    heldout_indices = set(validation_indices) | set(test_indices)
    train_indices = np.asarray(
        [index for index in range(len(mols)) if index not in heldout_indices],
        dtype=int,
    )

    validation_metrics = _cross_split_metrics(
        validation_indices, train_indices, similarities
    )
    test_metrics = _cross_split_metrics(test_indices, train_indices, similarities)
    if validation_metrics["mean_max"] >= max_mean_similarity:
        raise SimilarityConstrainedSplitError(
            "Validation-to-training mean maximum similarity failed validation"
        )
    if test_metrics["mean_max"] >= max_mean_similarity:
        raise SimilarityConstrainedSplitError(
            "Test-to-training mean maximum similarity failed validation"
        )
    if validation_metrics["maximum"] >= max_pair_similarity:
        raise SimilarityConstrainedSplitError(
            "Validation-to-training maximum similarity failed validation"
        )
    if test_metrics["maximum"] >= max_pair_similarity:
        raise SimilarityConstrainedSplitError(
            "Test-to-training maximum similarity failed validation"
        )

    scaffold_sets = {
        "training": {scaffolds[index] for index in train_indices},
        "validation": {scaffolds[index] for index in validation_indices},
        "test": {scaffolds[index] for index in test_indices},
    }
    shared_scaffolds = {
        "training_validation": len(
            scaffold_sets["training"] & scaffold_sets["validation"]
        ),
        "training_test": len(
            scaffold_sets["training"] & scaffold_sets["test"]
        ),
        "validation_test": len(
            scaffold_sets["validation"] & scaffold_sets["test"]
        ),
    }
    if any(shared_scaffolds.values()):
        raise SimilarityConstrainedSplitError(
            f"Scaffold overlap detected after optimization: {shared_scaffolds}"
        )

    diagnostics = {
        "algorithm": "similarity_constrained_scaffold_milp",
        "seed": seed,
        "fingerprint": "ECFP4",
        "fingerprint_bits": fingerprint_bits,
        "max_mean_similarity_limit": max_mean_similarity,
        "max_pair_similarity_limit": max_pair_similarity,
        "component_count": component_count,
        "split_counts": {
            "training": {
                "molecules": train_size,
                "positive": int(labels_array[train_indices].sum()),
                "negative": int(train_size - labels_array[train_indices].sum()),
            },
            "validation": {
                "molecules": validation_size,
                "positive": int(labels_array[validation_indices].sum()),
                "negative": int(
                    validation_size - labels_array[validation_indices].sum()
                ),
            },
            "test": {
                "molecules": test_size,
                "positive": int(labels_array[test_indices].sum()),
                "negative": int(test_size - labels_array[test_indices].sum()),
            },
        },
        "validation_to_training": validation_metrics,
        "test_to_training": test_metrics,
        "shared_scaffolds": shared_scaffolds,
        "solver_message": result.message,
    }
    return train_indices, validation_indices, test_indices, diagnostics
