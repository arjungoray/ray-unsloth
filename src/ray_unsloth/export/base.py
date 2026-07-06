"""Exporter registry and built-in local/HF/stub export workflows."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from ray_unsloth.checkpoints import MANIFEST_NAME, read_manifest, resolve_path
from ray_unsloth.errors import ExportError
from ray_unsloth.plugins import exporters as _registry


@dataclass(slots=True)
class ExportReport:
    target: str
    source_path: str
    output_path: str
    created_at: float
    artifacts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Exporter(Protocol):
    name: str

    def export(self, checkpoint_path: str, output: str | None = None, **options: Any) -> ExportReport: ...


def register_exporter(name: str, exporter: Exporter, *, description: str = "", replace: bool = False) -> Exporter:
    _registry.register(name, exporter, description=description, replace=replace)
    return exporter


def list_exporters() -> list[str]:
    return _registry.names()


def export_checkpoint(target: str, checkpoint_path: str, output: str | None = None, **options: Any) -> ExportReport:
    try:
        exporter = _registry.get(target)
    except Exception as exc:
        raise ExportError(f"Unknown export target '{target}'. Available: {', '.join(_registry.names())}.") from exc
    return cast(ExportReport, exporter.export(checkpoint_path, output=output, **options))


class LocalDirExporter:
    name = "local"

    def export(self, checkpoint_path: str, output: str | None = None, **options: Any) -> ExportReport:
        del options
        source = resolve_path(checkpoint_path)
        if not source.exists():
            raise ExportError(f"Checkpoint path does not exist: {source}")
        output_path = resolve_path(output or f"{source}-export")
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(source, output_path)
        manifest = read_manifest(output_path)
        created_at = time.time()
        export_manifest = {
            "export_target": self.name,
            "source_path": str(source),
            "checkpoint_manifest": manifest,
            "created_at": created_at,
        }
        (output_path / "export.json").write_text(json.dumps(export_manifest, indent=2, sort_keys=True))
        return ExportReport(
            target=self.name,
            source_path=str(source),
            output_path=str(output_path),
            created_at=created_at,
            artifacts=[MANIFEST_NAME, "export.json"],
            notes=["Copied checkpoint artifacts without merging weights."],
        )


class HuggingFaceExporter(LocalDirExporter):
    name = "hf"

    def export(self, checkpoint_path: str, output: str | None = None, **options: Any) -> ExportReport:
        report = super().export(checkpoint_path, output=output, **options)
        readme = Path(report.output_path) / "README.md"
        manifest = read_manifest(report.output_path)
        readme.write_text(
            "\n".join(
                [
                    "---",
                    "library_name: peft",
                    "tags:",
                    "- ray-unsloth",
                    "- lora",
                    "---",
                    "",
                    "# ray-unsloth checkpoint export",
                    "",
                    f"Base model: `{manifest.get('base_model', 'unknown')}`",
                    "",
                    "This export contains the checkpoint artifacts produced by ray-unsloth.",
                ]
            )
            + "\n"
        )
        report.target = self.name
        report.artifacts.append("README.md")
        report.notes.append("Prepared local Hugging Face-style folder; push with huggingface_hub if desired.")
        return report


class StubExporter:
    def __init__(self, name: str, instructions: list[str]):
        self.name = name
        self.instructions = instructions

    def export(self, checkpoint_path: str, output: str | None = None, **options: Any) -> ExportReport:
        del options
        source = resolve_path(checkpoint_path)
        if not source.exists():
            raise ExportError(f"Checkpoint path does not exist: {source}")
        output_path = resolve_path(output or f"{source}-{self.name}-plan")
        output_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "target": self.name,
            "source_path": str(source),
            "instructions": self.instructions,
            "created_at": time.time(),
        }
        (output_path / "export-plan.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        return ExportReport(
            target=self.name,
            source_path=str(source),
            output_path=str(output_path),
            created_at=float(cast(float, payload["created_at"])),
            artifacts=["export-plan.json"],
            notes=self.instructions,
        )


def _find_convert_script(name: str, llama_cpp_dir: str | None) -> Path | None:
    """Locate a llama.cpp convert script from options, env, or a sibling checkout."""
    import os

    candidates = []
    if llama_cpp_dir:
        candidates.append(Path(llama_cpp_dir).expanduser() / name)
    env_dir = os.environ.get("RAY_UNSLOTH_LLAMA_CPP")
    if env_dir:
        candidates.append(Path(env_dir).expanduser() / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class GGUFExporter:
    """Convert a saved LoRA adapter checkpoint to a GGUF LoRA.

    Runs llama.cpp's ``convert_lora_to_gguf.py`` when a checkout is available
    (via the ``llama_cpp_dir`` option or the ``RAY_UNSLOTH_LLAMA_CPP`` env
    var); otherwise emits a plan artifact with the exact commands.
    """

    name = "gguf"

    def export(self, checkpoint_path: str, output: str | None = None, **options: Any) -> ExportReport:
        import subprocess
        import sys

        source = resolve_path(checkpoint_path)
        if not source.exists():
            raise ExportError(f"Checkpoint path does not exist: {source}")
        manifest = read_manifest(source)
        base_model = str(options.get("base_model") or manifest.get("base_model") or "")
        outtype = str(options.get("outtype", "f16"))
        output_path = resolve_path(output or f"{source}-gguf")
        output_path.mkdir(parents=True, exist_ok=True)
        outfile = output_path / f"{source.name}-lora-{outtype}.gguf"

        script = _find_convert_script("convert_lora_to_gguf.py", options.get("llama_cpp_dir"))
        command = [
            sys.executable,
            str(script) if script else "convert_lora_to_gguf.py",
            str(source),
            "--outfile",
            str(outfile),
            "--outtype",
            outtype,
            "--base-model-id",
            base_model,
        ]
        if script is None:
            payload = {
                "target": self.name,
                "source_path": str(source),
                "instructions": [
                    "git clone --depth 1 https://github.com/ggml-org/llama.cpp && pip install gguf",
                    " ".join(command),
                    "Serve with a quantized base model: llama-cli -m <base.gguf> --lora <adapter>.gguf",
                ],
                "created_at": time.time(),
            }
            (output_path / "export-plan.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
            return ExportReport(
                target=self.name,
                source_path=str(source),
                output_path=str(output_path),
                created_at=float(cast(float, payload["created_at"])),
                artifacts=["export-plan.json"],
                notes=[
                    "llama.cpp checkout not found — wrote the conversion plan instead.",
                    "Set RAY_UNSLOTH_LLAMA_CPP or pass llama_cpp_dir=<checkout> to run the conversion.",
                ],
            )

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise ExportError(
                f"convert_lora_to_gguf.py failed (exit {result.returncode}):\n{result.stderr.strip()[-2000:]}"
            )
        return ExportReport(
            target=self.name,
            source_path=str(source),
            output_path=str(output_path),
            created_at=time.time(),
            artifacts=[outfile.name],
            notes=[
                f"Converted LoRA adapter to GGUF ({outtype}) for base model {base_model}.",
                "Serve with a quantized base: llama-cli -m <base.gguf> --lora <this adapter .gguf> (the adapter stays f16).",
            ],
        )


class OllamaExporter:
    """Produce an Ollama Modelfile for the adapter (GGUF conversion first).

    Writes ``FROM <base>`` + ``ADAPTER`` Modelfile next to the converted GGUF
    LoRA and, when the ``ollama`` CLI is available and ``create=True`` is
    passed, runs ``ollama create``.
    """

    name = "ollama"

    def export(self, checkpoint_path: str, output: str | None = None, **options: Any) -> ExportReport:
        import shutil as _shutil
        import subprocess

        source = resolve_path(checkpoint_path)
        manifest = read_manifest(source)
        base_model = str(options.get("base_model") or manifest.get("base_model") or "<base-model>")
        ollama_base = str(options.get("ollama_base") or base_model.split("/")[-1].lower())
        output_path = resolve_path(output or f"{source}-ollama")

        gguf_report = GGUFExporter().export(checkpoint_path, output=str(output_path), **options)
        gguf_files = [name for name in gguf_report.artifacts if name.endswith(".gguf")]
        adapter_line = f"ADAPTER ./{gguf_files[0]}" if gguf_files else "ADAPTER ./<adapter>.gguf"
        modelfile = output_path / "Modelfile"
        modelfile.write_text(f"FROM {ollama_base}\n{adapter_line}\n")

        model_name = str(options.get("model_name") or f"{source.name}-ray-unsloth")
        notes = list(gguf_report.notes)
        artifacts = [*gguf_report.artifacts, "Modelfile"]
        ollama_bin = _shutil.which("ollama")
        if ollama_bin is None:
            notes.append(f"ollama CLI not found — install it and run: ollama create {model_name} -f {modelfile}")
        elif options.get("create"):
            result = subprocess.run(
                [ollama_bin, "create", model_name, "-f", str(modelfile)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise ExportError(f"ollama create failed: {result.stderr.strip()[-2000:]}")
            notes.append(f"Created Ollama model '{model_name}'. Try: ollama run {model_name}")
        else:
            notes.append(f"Run: ollama create {model_name} -f {modelfile} (or pass create=True)")
        return ExportReport(
            target=self.name,
            source_path=str(source),
            output_path=str(output_path),
            created_at=time.time(),
            artifacts=artifacts,
            notes=notes,
        )


for _name, _exporter, _description in (
    ("local", LocalDirExporter(), "Copy checkpoint artifacts into a local export folder."),
    ("hf", HuggingFaceExporter(), "Prepare a local Hugging Face-style LoRA adapter folder."),
    ("gguf", GGUFExporter(), "Convert the LoRA adapter to GGUF via llama.cpp (plan artifact if tooling missing)."),
    (
        "ollama",
        OllamaExporter(),
        "GGUF-convert the adapter and write an Ollama Modelfile (ollama create with create=True).",
    ),
    (
        "vllm",
        StubExporter("vllm", ["Serve the base model with the exported LoRA adapter via vLLM --enable-lora."]),
        "Render vLLM serving plan.",
    ),
    (
        "sglang",
        StubExporter("sglang", ["Serve the base model and LoRA adapter with SGLang's LoRA support."]),
        "Render SGLang serving plan.",
    ),
):
    if _name not in _registry:
        register_exporter(_name, _exporter, description=_description)
