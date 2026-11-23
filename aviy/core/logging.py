"""Shared logging configuration and logger creation."""

import logging

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: int | str = "INFO") -> None:
    """Configure basic application logging once during startup.

    Applications should avoid logging credentials, secrets, private user
    content, or other sensitive information.
    """
    if isinstance(level, str):
        resolved_level = getattr(logging, level.upper(), None)
        if not isinstance(resolved_level, int):
            raise ValueError(f"Unknown logging level: {level}")  # noqa: TRY004
    elif isinstance(level, int) and not isinstance(level, bool):
        resolved_level = level
    else:
        raise TypeError("Logging level must be a level name or integer.")

    logging.basicConfig(
        level=resolved_level,
        format=_LOG_FORMAT,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger within the Aviy logger namespace."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Logger name must be a non-empty string.")

    qualified_name = name if name.startswith("aviy.") else f"aviy.{name}"
    return logging.getLogger(qualified_name)
