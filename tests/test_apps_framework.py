"""App-framework coverage via an in-test dummy app (real apps live in their own repos)."""

import argparse
import json

from ray_unsloth.apps import AppManifest, StageSpec, register_app
from ray_unsloth.cli import main as cli_main
from ray_unsloth.plugins import apps as apps_registry


def _dummy_manifest() -> AppManifest:
    def build_cli(subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser("run", help="Run the dummy app.")
        parser.set_defaults(func=lambda args: (print("dummy ran"), 0)[1])

    return AppManifest(
        name="dummyapp",
        description="A dummy app for framework tests.",
        stages=[StageSpec(name="one", description="first"), StageSpec(name="two", description="second")],
        build_cli=build_cli,
    )


def test_registered_app_appears_in_apps_listing_and_mounts_cli(capsys):
    if "dummyapp" not in apps_registry:
        register_app(_dummy_manifest())

    assert cli_main(["apps", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    dummy = next(row for row in rows if row["name"] == "dummyapp")
    assert [stage["name"] for stage in dummy["stages"]] == ["one", "two"]

    assert cli_main(["dummyapp", "run"]) == 0
    assert "dummy ran" in capsys.readouterr().out
