"""Modal image construction helpers.

Flash-attn and vLLM cannot share one Modal image in this runtime split:
flash-attn is pinned to a torch 2.8 / CUDA 12 wheel, while the vLLM path
forces torch 2.10 / CUDA 13.0. Keeping them together would make one of the two
dependency stacks invalid, so the image builder has to branch before package
installation.
"""

from __future__ import annotations

from ray_unsloth.config import ModelConfig, RuntimeConfig

VLLM_CU130_WHEEL = (
    "https://github.com/vllm-project/vllm/releases/download/v0.19.1/"
    "vllm-0.19.1+cu130-cp38-abi3-manylinux_2_35_x86_64.whl"
)
FLASH_ATTN_VERSION = "2.8.3"
FLASH_ATTN_TORCH_VERSION = "2.8"
FLASH_ATTN_WHEEL_BASE_URL = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3"
FLASH_LINEAR_ATTENTION_PACKAGE = "flash-linear-attention==0.5.0"
CAUSAL_CONV1D_VERSION = "1.6.1"
CAUSAL_CONV1D_TAG = "v1.6.1.post4"
CAUSAL_CONV1D_WHEEL_BASE_URL = "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4"


def _modal_requires_qwen3_5_transformers(model_config: ModelConfig) -> bool:
    normalized = model_config.base_model.lower().replace("-", "_").replace("/", "_").replace(".", "_")
    return "qwen3_5" in normalized


def _modal_any_requires_qwen3_5_transformers(config: RuntimeConfig) -> bool:
    return _modal_requires_qwen3_5_transformers(config.model) or any(
        _modal_requires_qwen3_5_transformers(model_config.model) for model_config in config.model_configs.values()
    )


def _modal_wants_vllm(model_config: ModelConfig) -> bool:
    if _modal_requires_qwen3_5_transformers(model_config):
        return False
    if model_config.fast_inference is True:
        return True
    return model_config.fast_inference == "auto" and not model_config.trust_remote_code


def _modal_any_wants_vllm(config: RuntimeConfig) -> bool:
    return _modal_wants_vllm(config.model) or any(
        _modal_wants_vllm(model_config.model) for model_config in config.model_configs.values()
    )


def _modal_wants_flash_attention_package(config: RuntimeConfig) -> bool:
    return not _modal_any_wants_vllm(config)


def _modal_torch_cuda_tags(config: RuntimeConfig) -> tuple[str, str]:
    if _modal_wants_flash_attention_package(config):
        return "2.8", "cu12"
    return "2.10", "cu13"


def _modal_transformers_package(config: RuntimeConfig) -> str:
    if _modal_any_requires_qwen3_5_transformers(config):
        return "transformers==5.5.0"
    if _modal_any_wants_vllm(config):
        return "transformers==4.57.6"
    return "transformers==5.5.0"


def _modal_huggingface_hub_package(config: RuntimeConfig) -> str:
    if _modal_transformers_package(config) == "transformers==5.5.0":
        return "huggingface_hub==1.14.0"
    return "huggingface_hub==0.36.2"


def _modal_torch_backend_packages(config: RuntimeConfig) -> list[str]:
    if _modal_wants_flash_attention_package(config):
        packages = [
            "torch==2.8.0",
            "torchvision==0.23.0",
            "xformers==0.0.32.post2",
            "packaging==26.2",
            "wheel==0.45.1",
        ]
    else:
        packages = [
            "torch==2.10.0",
            "torchvision==0.25.0",
            "xformers==0.0.34",
            "packaging==26.2",
            "wheel==0.45.1",
        ]
    if _modal_any_wants_vllm(config):
        packages.append(VLLM_CU130_WHEEL)
    return packages


def _modal_flash_attention_packages(config: RuntimeConfig) -> list[str]:
    if not _modal_wants_flash_attention_package(config):
        return []
    python_tag = config.modal.python_version.replace(".", "")
    wheel_name = (
        f"flash_attn-{FLASH_ATTN_VERSION}+cu12torch{FLASH_ATTN_TORCH_VERSION}"
        f"cxx11abiTRUE-cp{python_tag}-cp{python_tag}-linux_x86_64.whl"
    )
    wheel_url = f"{FLASH_ATTN_WHEEL_BASE_URL}/{wheel_name.replace('+', '%2B')}"
    return [f"flash-attn @ {wheel_url}"]


def _modal_causal_conv1d_package(config: RuntimeConfig) -> str:
    python_tag = config.modal.python_version.replace(".", "")
    torch_tag, cuda_tag = _modal_torch_cuda_tags(config)
    wheel_name = (
        f"causal_conv1d-{CAUSAL_CONV1D_VERSION}+{cuda_tag}torch{torch_tag}"
        f"cxx11abiTRUE-cp{python_tag}-cp{python_tag}-linux_x86_64.whl"
    )
    wheel_url = f"{CAUSAL_CONV1D_WHEEL_BASE_URL}/{wheel_name.replace('+', '%2B')}"
    return f"causal-conv1d @ {wheel_url}"


def _modal_linear_attention_packages(config: RuntimeConfig) -> list[str]:
    return [
        FLASH_LINEAR_ATTENTION_PACKAGE,
        _modal_causal_conv1d_package(config),
    ]


def _modal_base_image(modal, config: RuntimeConfig, flash_attention_packages: list[str]):
    del flash_attention_packages
    modal_config = config.modal
    return modal.Image.debian_slim(python_version=modal_config.python_version)


def _modal_python_packages(config: RuntimeConfig) -> list[str]:
    return [
        "accelerate==1.13.0",
        "bitsandbytes==0.49.2",
        "hf-transfer==0.1.9",
        _modal_huggingface_hub_package(config),
        "peft==0.19.1",
        "ray==2.55.1",
        "pyyaml==6.0.3",
        _modal_transformers_package(config),
        "trl==0.24.0",
        "unsloth[colab-new]>=2026.5.2",
        "unsloth-zoo>=2026.5.1",
    ]
