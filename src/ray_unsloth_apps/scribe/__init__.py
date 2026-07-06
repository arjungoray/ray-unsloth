"""Scribe: a personal writing-voice training app."""

from __future__ import annotations

from ray_unsloth.apps import AppManifest, StageSpec, register_app
from ray_unsloth_apps.scribe.cli import build_cli

MANIFEST = AppManifest(
    name="scribe",
    description="Personal writing-voice model pipeline (ingest → profile → sft → rl → eval → export).",
    stages=[
        StageSpec(name="ingest", description="Ingest text and markdown files into the corpus store."),
        StageSpec(name="profile", description="Build the style profile and stylometric index."),
        StageSpec(name="sft", description="Run supervised fine-tuning on prompt→passage pairs."),
        StageSpec(name="rl", description="Run GRPO-style policy refinement against the style rubric."),
        StageSpec(name="eval", description="Evaluate the current checkpoint against the held-out corpus."),
        StageSpec(name="export", description="Export the latest checkpoint to a serving target."),
    ],
    build_cli=build_cli,
)


def register() -> None:
    register_app(MANIFEST)


register()
