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
    graph_cls: Type[Graph] | None = None,
) -> Graph:
    """Convert input text chunks into a similarity graph."""

    graph_type = Graph if graph_cls is None else graph_cls
    graph = graph_type()

    n = len(chunks)
    if n == 0:
        return graph

    embeddings = get_embeddings(chunks)
    sim = cosine_similarity_matrix(embeddings)

    for idx in range(n):
        graph.add_node(idx)

    for i in range(n):
        sim[i, i] = 0.0
        for j in range(i + 1, n):
            if sim[i, j] >= tau:
                graph.add_edge(i, j, float(sim[i, j]))

        if min_edges_per_node > 0 and not graph.neighbors(i):
            j_star = int(np.argmax(sim[i]))
            best = float(sim[i, j_star])
            if best > 0:
                graph.add_edge(i, j_star, best)

    return graph



