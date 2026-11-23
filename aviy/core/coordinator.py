"""Component lifecycle coordination."""

from __future__ import annotations

from enum import Enum

from aviy.core.errors import DuplicateRegistrationError, LifecycleError
from aviy.core.interfaces import Component


class LifecycleState(Enum):
    """Coordinator lifecycle states."""

    NEW = "new"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


class Coordinator:
    """Coordinate initialization, startup and reverse-order shutdown."""

    def __init__(self) -> None:
        self._components: list[Component] = []
        self._initialized: list[Component] = []
        self._started: list[Component] = []
        self._state = LifecycleState.NEW

    @property
    def state(self) -> LifecycleState:
        """Return current coordinator state."""
        return self._state

    @property
    def component_names(self) -> tuple[str, ...]:
        """Return registered component names."""
        return tuple(component.name for component in self._components)

    def add(self, component: Component) -> None:
        """Register a component before initialization."""
        if self._state is not LifecycleState.NEW:
            raise LifecycleError(
                "Components can only be added before initialization."
            )

        required = ("initialize", "start", "stop")
        name = getattr(component, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Component name must be non-empty.")
        if any(not callable(getattr(component, item, None)) for item in required):
            raise TypeError("Component does not implement the lifecycle contract.")
        if any(existing.name == name for existing in self._components):
            raise DuplicateRegistrationError(
                f"Component already registered: {name}"
            )

        self._components.append(component)

    def initialize(self) -> None:
        """Initialize components in registration order."""
        if self._state is not LifecycleState.NEW:
            raise LifecycleError("Coordinator is not in the new state.")

        failed_name = "<unknown>"
        try:
            for component in self._components:
                failed_name = component.name
                component.initialize()
                self._initialized.append(component)
        except Exception as exc:
            self._cleanup()
            self._state = LifecycleState.FAILED
            raise LifecycleError(
                f"Component initialization failed: {failed_name}"
            ) from exc

        self._state = LifecycleState.INITIALIZED

    def start(self) -> None:
        """Start initialized components in registration order."""
        if self._state is not LifecycleState.INITIALIZED:
            raise LifecycleError("Coordinator must be initialized before startup.")

        failed_name = "<unknown>"
        try:
            for component in self._initialized:
                failed_name = component.name
                component.start()
                self._started.append(component)
        except Exception as exc:
            self._cleanup()
            self._state = LifecycleState.FAILED
            raise LifecycleError(
                f"Component startup failed: {failed_name}"
            ) from exc

        self._state = LifecycleState.RUNNING

    def shutdown(self) -> None:
        """Stop components in reverse initialization order."""
        if self._state is LifecycleState.STOPPED:
            return
        if self._state is LifecycleState.NEW:
            self._state = LifecycleState.STOPPED
            return

        failures = self._cleanup()
        if failures:
            self._state = LifecycleState.FAILED
            raise LifecycleError(
                f"{len(failures)} component(s) failed during shutdown."
            ) from failures[0]

        self._state = LifecycleState.STOPPED

    def _cleanup(self) -> list[Exception]:
        failures = []
        for component in reversed(self._initialized):
            try:
                component.stop()
            except Exception as exc:  # noqa: BLE001
                # Cleanup must collect arbitrary component shutdown failures.
                failures.append(exc)
        self._started.clear()
        self._initialized.clear()
        return failures
