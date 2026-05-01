"""Small exception compatibility layer for Tinker examples."""

from ray_unsloth.errors import RayUnslothError as TinkerError
from ray_unsloth.errors import RayUnavailableError, UnsupportedLossError

APIError = TinkerError
APIConnectionError = TinkerError
APITimeoutError = TinkerError
APIResponseValidationError = TinkerError
APIStatusError = TinkerError
AuthenticationError = TinkerError
BadRequestError = TinkerError
ConflictError = TinkerError
InternalServerError = TinkerError
NotFoundError = TinkerError
PermissionDeniedError = TinkerError
RateLimitError = TinkerError
RequestFailedError = TinkerError
SidecarDiedError = TinkerError
SidecarError = TinkerError
SidecarIPCError = TinkerError
SidecarStartupError = TinkerError
UnprocessableEntityError = TinkerError

__all__ = [name for name in globals() if not name.startswith("_")]
