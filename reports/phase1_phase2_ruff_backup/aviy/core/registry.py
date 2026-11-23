"""Validated, thread-safe component registry."""

from __future__ import annotations

from threading import RLock
from typing import Any, cast

from aviy.core.errors import (
    DuplicateRegistrationError,
    RegistryError,
    UnknownRegistrationError,
)

class Registry[T]:
    """Store and retrieve named components or component factories."""

    def __init__(self, name: str = "registry") -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Registry name must be non-empty.")
        self.name = name
        self._items: dict[str, T] = {}
        self._frozen = False
        self._lock = RLock()

    def register(
        self,
        name: str,
        value: T,
        *,
        replace: bool = False,
    ) -> T:
        """Register a value under a validated name."""
        self._validate_name(name)
        if value is None:
            raise ValueError("Registry values cannot be None.")

        with self._lock:
            if self._frozen:
                raise RegistryError("Registry is frozen.")
            if name in self._items and not replace:
                raise DuplicateRegistrationError(
                    f"Name is already registered: {name}"
                )
            self._items[name] = value
        return value

    def get(self, name: str) -> T:
        """Retrieve a registered value."""
        self._validate_name(name)
        with self._lock:
            try:
                return self._items[name]
            except KeyError as exc:
                raise UnknownRegistrationError(
                    f"Unknown registry name: {name}"
                ) from exc

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Instantiate a registered callable."""
        factory = self.get(name)
        if not callable(factory):
            raise RegistryError("Registered value is not callable.")
        return cast(Any, factory)(*args, **kwargs)

    def unregister(self, name: str) -> T:
        """Remove and return a registered value."""
        self._validate_name(name)
        with self._lock:
            if self._frozen:
                raise RegistryError("Registry is frozen.")
            try:
                return self._items.pop(name)
            except KeyError as exc:
                raise UnknownRegistrationError(
                    f"Unknown registry name: {name}"
                ) from exc

    def freeze(self) -> None:
        """Prevent further registration changes."""
        with self._lock:
            self._frozen = True

    @property
    def frozen(self) -> bool:
        """Return whether registry modifications are disabled."""
        with self._lock:
            return self._frozen

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in insertion order."""
        with self._lock:
            return tuple(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Registry name must be non-empty.")
