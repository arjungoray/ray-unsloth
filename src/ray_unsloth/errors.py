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


class PluginError(RayUnslothError):
    """Raised when a plugin cannot be found, loaded, or registered."""


class ProviderError(RayUnslothError):
    """Raised when a runtime provider fails to launch or manage a session."""


class ProviderNotAvailableError(ProviderError):
    """Raised when a provider exists but cannot run in this environment.

    Carries actionable guidance: which package to install, which CLI to set
    up, or a docs link for providers that only support planning today.
    """

    def __init__(self, message: str, *, hint: str | None = None):
        if hint:
            message = f"{message}\nHint: {hint}"
        super().__init__(message)
        self.hint = hint


class EvalError(RayUnslothError):
    """Raised when an eval run cannot be constructed or executed."""


class ExportError(RayUnslothError):
    """Raised when a checkpoint export fails."""
