"""Dependency injection container with explicit lifetimes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import RLock, local
from typing import Any

from aviy.core.errors import (
    CircularDependencyError,
    DependencyResolutionError,
    DuplicateRegistrationError,
)


class Lifetime(Enum):
    """Supported dependency lifetimes."""

    TRANSIENT = "transient"
    SINGLETON = "singleton"


@dataclass
class _Binding:
    factory: Callable[[Container], Any]
    lifetime: Lifetime
    instance: Any = None
    has_instance: bool = False


class Container:
    """Resolve dependencies through explicit provider registrations."""

    def __init__(self) -> None:
        self._bindings: dict[str, _Binding] = {}
        self._lock = RLock()
        self._local = local()

    def bind(
        self,
        key: str,
        factory: Callable[[Container], Any],
        *,
        lifetime: Lifetime = Lifetime.TRANSIENT,
        replace: bool = False,
    ) -> None:
        """Register a dependency provider."""
        self._validate_key(key)
        if not callable(factory):
            raise TypeError("Dependency factory must be callable.")
        if not isinstance(lifetime, Lifetime):
            raise TypeError("Lifetime must be a Lifetime value.")

        with self._lock:
            if key in self._bindings and not replace:
                raise DuplicateRegistrationError(f"Dependency already bound: {key}")
            self._bindings[key] = _Binding(factory, lifetime)

    def bind_instance(
        self,
        key: str,
        instance: Any,
        *,
        replace: bool = False,
    ) -> None:
        """Bind an existing object as a singleton instance."""
        self._validate_key(key)
        if instance is None:
            raise ValueError("Dependency instance cannot be None.")

        with self._lock:
            if key in self._bindings and not replace:
                raise DuplicateRegistrationError(f"Dependency already bound: {key}")
            self._bindings[key] = _Binding(
                factory=lambda _container: instance,
                lifetime=Lifetime.SINGLETON,
                instance=instance,
                has_instance=True,
            )

    def resolve(self, key: str) -> Any:
        """Resolve a dependency or raise a descriptive exception."""
        self._validate_key(key)

        stack = getattr(self._local, "resolution_stack", [])
        if key in stack:
            chain = " -> ".join([*stack, key])
            raise CircularDependencyError(f"Circular dependency detected: {chain}")

        with self._lock:
            binding = self._bindings.get(key)
            if binding is None:
                raise DependencyResolutionError(f"Dependency is not registered: {key}")

            if binding.lifetime is Lifetime.SINGLETON and binding.has_instance:
                return binding.instance

            self._local.resolution_stack = [*stack, key]
            try:
                instance = binding.factory(self)
                if instance is None:
                    raise DependencyResolutionError(
                        "Dependency provider returned None."
                    )
                if binding.lifetime is Lifetime.SINGLETON:
                    binding.instance = instance
                    binding.has_instance = True
                return instance
            except DependencyResolutionError:
                raise
            except Exception as exc:
                raise DependencyResolutionError(
                    f"Failed to resolve dependency: {key}"
                ) from exc
            finally:
                self._local.resolution_stack = stack

    def contains(self, key: str) -> bool:
        """Return whether a dependency is registered."""
        self._validate_key(key)
        with self._lock:
            return key in self._bindings

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Dependency key must be non-empty.")
