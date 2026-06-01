"""Compatibility re-exports for `tinker.lib.public_interfaces`.

The real Tinker SDK exposes its awaitable future type as
`tinker.lib.public_interfaces.APIFuture`. ray-unsloth's equivalent is
`AsyncMethodFuture`, the generic future returned by the async client methods.
"""

from ray_unsloth.types import AsyncMethodFuture as APIFuture

__all__ = ["APIFuture"]
