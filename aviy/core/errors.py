"""Shared exception hierarchy for Aviy."""


class AviyError(Exception):
    """Base class for expected Aviy application errors."""


class ConfigurationError(AviyError):
    """Invalid or incomplete application configuration."""


class ValidationError(AviyError):
    """Invalid input or data that fails validation."""


class ResourceLimitError(AviyError):
    """An operation exceeds an authorized resource limit."""


class SecurityError(AviyError):
    """A security or authorization requirement was not satisfied."""


class RecoveryError(AviyError):
    """A recovery or restoration operation failed."""


class RegistryError(AviyError):
    """A registry operation failed."""


class DuplicateRegistrationError(RegistryError):
    """A registry name is already registered."""


class UnknownRegistrationError(RegistryError):
    """A requested registry name does not exist."""


class DependencyResolutionError(AviyError):
    """A dependency could not be resolved."""


class CircularDependencyError(DependencyResolutionError):
    """A circular dependency was detected."""


class LifecycleError(AviyError):
    """A component lifecycle operation failed."""
