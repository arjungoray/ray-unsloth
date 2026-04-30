"""Project-specific exceptions."""


class RayUnslothError(Exception):
    """Base exception for ray-unsloth."""


class ConfigurationError(RayUnslothError):
    """Raised when runtime configuration is invalid."""


class RayUnavailableError(RayUnslothError):
    """Raised when Ray is required but unavailable."""


class UnsupportedLossError(RayUnslothError):
    """Raised when a requested loss is outside the non-RL MVP surface."""


class CheckpointError(RayUnslothError):
    """Raised when checkpoint save/load fails validation."""
