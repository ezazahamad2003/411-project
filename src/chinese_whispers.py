"""Chinese Whispers clustering algorithm implementation."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict
import random

from .graph import Graph


def chinese_whispers(
    graph: Graph, max_iterations: int = 20, seed: int | None = 42
) -> Dict[int, int]:
    """Run Chinese Whispers over the supplied graph.

    Args:
        graph: Undirected weighted graph describing pairwise similarities.
        max_iterations: Upper bound on label-propagation passes.
        seed: Optional RNG seed for deterministic shuffling.

    Returns:
        Mapping of node -> cluster label.

    Complexity:
        Each iteration visits every vertex and aggregates neighbor weights,
        resulting in O(|V| + |E|) work per pass. With k iterations (usually
        small in practice), total complexity is O(k * (|V| + |E|)).
    """

    nodes = graph.nodes()
    labels: Dict[int, int] = {node: node for node in nodes}
    rng = random.Random(seed)

    for _ in range(max_iterations):
        order = nodes[:]
        rng.shuffle(order)
        changes = 0

        for node in order:
            neighbors = graph.neighbors(node)
            if not neighbors:
                continue

            weight_by_label: Dict[int, float] = defaultdict(float)
            for neighbor_idx, weight in neighbors:
                weight_by_label[labels[neighbor_idx]] += weight

            best_label = max(
                weight_by_label.items(), key=lambda item: (item[1], -item[0])
            )[0]

            if best_label != labels[node]:
                labels[node] = best_label
                changes += 1

        if changes == 0:
            break

    return labels



