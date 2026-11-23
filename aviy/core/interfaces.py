"""Stable public interfaces for Aviy components."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Component(Protocol):
    """Contract implemented by a managed Aviy component."""

    @property
    def name(self) -> str:
        """Return the stable component name."""
        ...

    def initialize(self) -> None:
        """Prepare component resources."""
        ...

    def start(self) -> None:
        """Start component operation."""
        ...

    def stop(self) -> None:
        """Stop operation and release component resources."""
        ...
