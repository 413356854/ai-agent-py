"""Project specific exceptions."""

from __future__ import annotations


class AiAgentError(Exception):
    """Base class for all errors raised by this package."""


class ConfigurationError(AiAgentError):
    """Missing or invalid configuration (usually a missing API key)."""


class DocumentLoadError(AiAgentError):
    """A file could not be loaded or its type is not supported."""
