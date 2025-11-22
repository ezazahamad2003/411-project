"""Command-line interface for the Chinese Whispers pipeline."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .chinese_whispers import chinese_whispers
from .graph import Graph
from .embeddings import build_graph_from_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster text chunks with OpenAI embeddings + Chinese Whispers."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="data/sample_chunks.txt",
        help="Path to chunks file (one text chunk per line).",
    )
    parser.add_argument(
        "-t",
        "--tau",
        type=float,
        default=0.3,
        help="Cosine similarity threshold for edge creation.",
    )
    parser.add_argument(
        "-k",
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum Chinese Whispers iterations.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output/clusters.json",
        help="Where to write the JSON summary.",
    )
    parser.add_argument(
        "--min-edges-per-node",
        type=int,
        default=1,
        help="Ensure each node has at least this many edges by linking to top neighbors.",
    )
    return parser.parse_args()


def read_chunks(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def invert_labels(labels: Dict[int, int]) -> Dict[int, List[int]]:
    clusters: Dict[int, List[int]] = defaultdict(list)
    for node, label in labels.items():
        clusters[label].append(node)
    for members in clusters.values():
        members.sort()
    return clusters


def print_summary(chunks: List[str], graph: Graph, clusters: Dict[int, List[int]]) -> None:
    total_nodes = len(chunks)
    total_edges = graph.edge_count()
    print(f"Nodes: {total_nodes}")
    print(f"Edges: {total_edges}")
    print(f"Clusters: {len(clusters)}")

    sorted_clusters = sorted(
        clusters.items(), key=lambda item: len(item[1]), reverse=True
    )
    for label, members in sorted_clusters:
        sample_texts = [chunks[idx] for idx in members[:3]]
        print(f"- Cluster {label} | size {len(members)}")
        for text in sample_texts:
            print(f"    · {text}")


def write_json(
    path: Path,
    tau: float,
    max_iterations: int,
    graph: Graph,
    clusters: Dict[int, List[int]],
    chunks: List[str],
) -> None:
    sorted_clusters = sorted(
        clusters.items(), key=lambda item: len(item[1]), reverse=True
    )
    payload = {
        "tau": tau,
        "max_iterations": max_iterations,
        "num_nodes": len(chunks),
        "num_edges": graph.edge_count(),
        "clusters": [
            {
                "label": int(label),
                "node_indices": members,
                "texts": [chunks[idx] for idx in members],
            }
            for label, members in sorted_clusters
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    chunk_path = Path(args.input)
    chunks = read_chunks(chunk_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunk_path}")

    graph = build_graph_from_chunks(
        chunks, tau=args.tau, min_edges_per_node=args.min_edges_per_node
    )
    labels = chinese_whispers(graph, max_iterations=args.max_iterations)
    clusters = invert_labels(labels)

    print_summary(chunks, graph, clusters)
    write_json(Path(args.output), args.tau, args.max_iterations, graph, clusters, chunks)


if __name__ == "__main__":
    main()



