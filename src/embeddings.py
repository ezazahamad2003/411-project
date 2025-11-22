"""Embedding helpers and graph construction utilities."""

from __future__ import annotations

import os
from typing import Sequence, Type

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI
from dotenv import load_dotenv

from .graph import Graph

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY missing. Create a .env file based on .env.example."
    )

EMBED_MODEL = "text-embedding-3-small"
_client = OpenAI()


def get_embeddings(chunks: Sequence[str]) -> NDArray[np.float32]:
    """Call OpenAI's embeddings API and return a 2D numpy array."""
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)

    response = _client.embeddings.create(model=EMBED_MODEL, input=list(chunks))
    vectors = [item.embedding for item in response.data]
    return np.asarray(vectors, dtype=np.float32)


def cosine_similarity_matrix(X: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return cosine similarities for all row pairs in X."""
    if X.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X_norm = X / norms
    sim = X_norm @ X_norm.T
    return np.clip(sim, -1.0, 1.0).astype(np.float32)


def build_graph_from_chunks(
    chunks: Sequence[str],
    tau: float,
    min_edges_per_node: int = 0,
    top_k: int | None = None,
    graph_cls: Type[Graph] | None = None,
) -> Graph:
    """Convert input text chunks into a similarity graph.

    Args:
        chunks: Text snippets to embed.
        tau: Cosine similarity threshold for edge creation.
        min_edges_per_node: Fallback minimum degree via strongest neighbor.
        top_k: Optional cap on the number of outgoing edges per node.
        graph_cls: Dependency injection hook for testing.
    """

    graph_type = Graph if graph_cls is None else graph_cls
    graph = graph_type()

    n = len(chunks)
    if n == 0:
        return graph

    embeddings = get_embeddings(chunks)
    sim = cosine_similarity_matrix(embeddings)

    for idx in range(n):
        graph.add_node(idx)

    limit = None if top_k is None or top_k <= 0 else int(top_k)

    if limit is None:
        for i in range(n):
            sim[i, i] = 0.0
            row = sim[i]
            for j in range(i + 1, n):
                if row[j] >= tau:
                    graph.add_edge(i, j, float(row[j]))

            if min_edges_per_node > 0 and not graph.neighbors(i):
                j_star = int(np.argmax(row))
                best = float(row[j_star])
                if best > 0:
                    graph.add_edge(i, j_star, best)
        return graph

    for i in range(n):
        row = sim[i].copy()
        row[i] = 0.0
        sorted_idx = np.argsort(-row)
        edges_added = 0

        for j in sorted_idx:
            if row[j] < tau:
                break
            graph.add_edge(i, int(j), float(row[j]))
            edges_added += 1
            if edges_added >= limit:
                break

        if min_edges_per_node > 0 and not graph.neighbors(i):
            j_star = int(np.argmax(row))
            best = float(row[j_star])
            if best > 0:
                graph.add_edge(i, j_star, best)

    return graph



