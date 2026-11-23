"""Graph and vector retrieval interfaces with deterministic local backends."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol


class GraphRetrieval(Protocol):
    def add_node(self, node_id: str, attributes: dict[str, str]) -> None: ...
    def add_edge(self, source: str, target: str, relation: str) -> None: ...
    def related(self, node_id: str, relation: str | None = None) -> list[str]: ...


class VectorRetrieval(Protocol):
    def upsert(self, item_id: str, vector: Sequence[float]) -> None: ...
    def query(
        self, vector: Sequence[float], limit: int = 5
    ) -> list[tuple[str, float]]: ...


class InMemoryGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, str]] = {}
        self.edges: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def add_node(self, node_id: str, attributes: dict[str, str]) -> None:
        if not node_id.strip():
            raise ValueError("node_id cannot be empty")
        self.nodes[node_id] = dict(attributes)

    def add_edge(self, source: str, target: str, relation: str) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise KeyError("Both graph nodes must exist before adding an edge")
        if not relation.strip():
            raise ValueError("relation cannot be empty")
        edge = (target, relation)
        if edge not in self.edges[source]:
            self.edges[source].append(edge)

    def related(self, node_id: str, relation: str | None = None) -> list[str]:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        return [
            target for target, edge_relation in self.edges[node_id]
            if relation is None or edge_relation == relation
        ]


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._vectors: dict[str, tuple[float, ...]] = {}

    @staticmethod
    def _validate(vector: Sequence[float]) -> tuple[float, ...]:
        values = tuple(float(value) for value in vector)
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("Vectors must be non-empty and finite")
        return values

    def upsert(self, item_id: str, vector: Sequence[float]) -> None:
        if not item_id.strip():
            raise ValueError("item_id cannot be empty")
        values = self._validate(vector)
        if self._vectors:
            expected = len(next(iter(self._vectors.values())))
            if len(values) != expected:
                raise ValueError("Vector dimensions do not match the index")
        self._vectors[item_id] = values

    def query(
        self, vector: Sequence[float], limit: int = 5
    ) -> list[tuple[str, float]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query = self._validate(vector)
        results = []
        for item_id, candidate in self._vectors.items():
            if len(candidate) != len(query):
                raise ValueError("Query vector dimension does not match index")
            dot = sum(left * right for left, right in zip(query, candidate))
            norm_q = math.sqrt(sum(value * value for value in query))
            norm_c = math.sqrt(sum(value * value for value in candidate))
            similarity = dot / (norm_q * norm_c) if norm_q and norm_c else 0.0
            results.append((item_id, similarity))
        return sorted(results, key=lambda item: (-item[1], item[0]))[:limit]
