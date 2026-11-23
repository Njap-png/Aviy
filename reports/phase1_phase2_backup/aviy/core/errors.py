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
