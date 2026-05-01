"""Compatibility re-exports for `import tinker.types`.

The real Tinker SDK exposes generated type modules such as
`tinker.types.tensor_data`. A few cookbook examples import those directly, so
this module also registers lightweight submodule shims.
"""

from ray_unsloth.types import *  # noqa: F403
import sys
import types as _module_types


__path__ = []  # Let importlib treat this compatibility module like a package.


def _register_type_module(module_name: str, **symbols) -> None:
    module = _module_types.ModuleType(f"{__name__}.{module_name}")
    module.__dict__.update(symbols)
    module.__all__ = list(symbols)
    sys.modules[module.__name__] = module


_register_type_module("tensor_data", TensorData=TensorData)  # noqa: F405
_register_type_module("model_input", ModelInput=ModelInput)  # noqa: F405
_register_type_module("encoded_text_chunk", EncodedTextChunk=EncodedTextChunk)  # noqa: F405
_register_type_module("image_chunk", ImageChunk=ImageChunk)  # noqa: F405
_register_type_module("image_asset_pointer_chunk", ImageAssetPointerChunk=ImageAssetPointerChunk)  # noqa: F405
_register_type_module("datum", Datum=Datum)  # noqa: F405
_register_type_module("sampling_params", SamplingParams=SamplingParams)  # noqa: F405
_register_type_module("optim_step_request", AdamParams=AdamParams)  # noqa: F405
_register_type_module("optim_step_response", OptimStepResponse=OptimStepResponse)  # noqa: F405
_register_type_module("sample_response", SampleResponse=SampleResponse)  # noqa: F405
_register_type_module("sampled_sequence", SampledSequence=SampledSequence)  # noqa: F405
_register_type_module("forward_backward_output", ForwardBackwardOutput=ForwardBackwardOutput)  # noqa: F405

__all__ = [name for name in globals() if not name.startswith("_")]
