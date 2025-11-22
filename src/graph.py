"""Undirected weighted graph utilities used by Chinese Whispers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

Neighbor = Tuple[int, float]


class Graph:
    """Simple adjacency-list graph implementation."""

    def __init__(self) -> None:
        # neighbors[node] -> List[(neighbor, weight)]
        self._adjacency: Dict[int, List[Neighbor]] = {}
        self._edges: set[tuple[int, int]] = set()

    def add_node(self, node_id: int) -> None:
        """Ensure the node exists in the adjacency list."""
        if node_id not in self._adjacency:
            self._adjacency[node_id] = []

    def add_edge(self, u: int, v: int, weight: float) -> None:
        """Insert an undirected weighted edge."""
        if u == v:
            return

        self.add_node(u)
        self.add_node(v)

        self._upsert_neighbor(u, v, weight)
        self._upsert_neighbor(v, u, weight)
        edge_key = (min(u, v), max(u, v))
        self._edges.add(edge_key)

    def _upsert_neighbor(self, u: int, v: int, weight: float) -> None:
        neighbors = self._adjacency[u]
        for idx, (neighbor_id, _) in enumerate(neighbors):
            if neighbor_id == v:
                neighbors[idx] = (v, weight)
                return
        neighbors.append((v, weight))

    def neighbors(self, u: int) -> List[Neighbor]:
        """Return the adjacency list for node u."""
        return list(self._adjacency.get(u, []))

    def nodes(self) -> List[int]:
        """List all node ids currently in the graph."""
        return list(self._adjacency.keys())

    def edge_count(self) -> int:
        """Return the number of undirected edges."""
        return len(self._edges)

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self._adjacency)

    def __iter__(self) -> Iterable[int]:  # pragma: no cover - convenience
        return iter(self._adjacency)


