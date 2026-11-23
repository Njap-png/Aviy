"""Shared Aviy core services."""

from aviy.core.container import Container, Lifetime
from aviy.core.coordinator import Coordinator, LifecycleState
from aviy.core.errors import (
    AviyError,
    CircularDependencyError,
    ConfigurationError,
    DependencyResolutionError,
    DuplicateRegistrationError,
    LifecycleError,
    RecoveryError,
    RegistryError,
    ResourceLimitError,
    SecurityError,
    UnknownRegistrationError,
    ValidationError,
)
from aviy.core.interfaces import Component
from aviy.core.logging import configure_logging, get_logger
from aviy.core.registry import Registry

__all__ = [
    "AviyError",
    "CircularDependencyError",
    "Component",
    "ConfigurationError",
    "Container",
    "Coordinator",
    "DependencyResolutionError",
    "DuplicateRegistrationError",
    "LifecycleError",
    "LifecycleState",
    "Lifetime",
    "RecoveryError",
    "Registry",
    "RegistryError",
    "ResourceLimitError",
    "SecurityError",
    "UnknownRegistrationError",
    "ValidationError",
    "configure_logging",
    "get_logger",
]
