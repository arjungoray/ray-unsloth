"""Project-specific exceptions and stable error codes."""

from __future__ import annotations

ERROR_DOCS_BASE = "https://arjungoray.github.io/ray-unsloth/reference/errors"

ERROR_CATALOG: dict[str, str] = {
    "RU-1001": "Invalid speed.profile value.",
    "RU-1002": "Invalid speed.optimizer value.",
    "RU-1003": "Invalid speed flag value.",
    "RU-1004": "Invalid distributed.mode value.",
    "RU-1005": "Invalid distributed.num_nodes value.",
    "RU-1006": "Invalid distributed.gpus_per_node value.",
    "RU-1007": "Conflicting provider and modal.enabled settings.",
    "RU-1008": "Invalid model config mapping.",
    "RU-1009": "Unknown default model config alias.",
    "RU-1010": "Invalid config field name or value.",
    "RU-2001": "Checkpoint path does not exist.",
    "RU-2002": "Missing checkpoint manifest.",
    "RU-2003": "Malformed checkpoint manifest JSON.",
    "RU-2004": "Malformed checkpoint manifest object.",
    "RU-2005": "Checkpoint base_model mismatch.",
    "RU-2006": "Checkpoint lora.rank mismatch.",
    "RU-2007": "Checkpoint lora.target_modules mismatch.",
    "RU-2008": "Checkpoint is missing fake-engine weights.",
    "RU-3001": "Unknown or invalid provider registration.",
    "RU-3002": "Provider cannot execute sessions in this build.",
    "RU-3003": "Ray is unavailable for local orchestration.",
    "RU-3004": "Ray is unavailable for Modal orchestration.",
    "RU-4001": "Unsupported loss name.",
    "RU-4002": "Datum is missing required loss inputs.",
    "RU-4003": "Unsupported policy-gradient loss.",
    "RU-4004": "Custom loss is not registered.",
    "RU-4005": "Non-finite training loss.",
    "RU-4006": "Training batch is empty.",
    "RU-4007": "Invalid stop-sequence tokenization.",
    "RU-4008": "Rank 0 did not produce a checkpoint manifest.",
    "RU-5001": "Unknown export target.",
    "RU-5002": "Checkpoint path does not exist for export.",
    "RU-5003": "GGUF conversion failed.",
    "RU-5004": "Ollama create failed.",
    "RU-6001": "Invalid lazy plugin reference.",
    "RU-6002": "Plugin module import failed.",
    "RU-6003": "Plugin attribute lookup failed.",
    "RU-6004": "Plugin name already registered.",
    "RU-6005": "Unknown plugin registry entry.",
}


def docs_url_for_code(code: str) -> str:
    return f"{ERROR_DOCS_BASE}#{code.lower()}"


class RayUnslothError(Exception):
    """Base exception for ray-unsloth."""

    code: str | None
    hint: str | None

    def __init__(self, message: str = "", *, code: str | None = None, hint: str | None = None):
        super().__init__(message)
        self.code = code
        self.hint = hint

    def __str__(self) -> str:
        message = super().__str__()
        if not self.code:
            return message
        lines = [f"{self.code}: {message}"]
        if self.hint:
            lines.append(f"Hint: {self.hint}")
        lines.append(f"Docs: {docs_url_for_code(self.code)}")
        return "\n".join(lines)


class ConfigurationError(RayUnslothError, ValueError):
    """Raised when runtime configuration is invalid."""


class RayUnavailableError(RayUnslothError):
    """Raised when Ray is required but unavailable."""


class UnsupportedLossError(RayUnslothError):
    """Raised when a requested loss is outside the non-RL MVP surface."""


class TrainingError(RayUnslothError):
    """Raised when a training step or training-side lifecycle action fails."""


class CheckpointError(RayUnslothError):
    """Raised when checkpoint save/load fails validation."""


class PluginError(RayUnslothError):
    """Raised when a plugin cannot be found, loaded, or registered."""


class ProviderError(RayUnslothError):
    """Raised when a runtime provider fails to launch or manage a session."""


class ProviderNotAvailableError(ProviderError):
    """Raised when a provider exists but cannot run in this environment."""

    def __init__(self, message: str, *, code: str | None = None, hint: str | None = None):
        super().__init__(message, code=code, hint=hint)


class EvalError(RayUnslothError):
    """Raised when an eval run cannot be constructed or executed."""


class ExportError(RayUnslothError):
    """Raised when a checkpoint export fails."""
