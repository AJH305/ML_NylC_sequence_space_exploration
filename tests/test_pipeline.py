from copy import deepcopy
import json
import shutil
import pytest
from nylc.runner import Runner
from nylc.config import load_config
from conftest import ROOT


def isolated_project(tmp_path):
    for folder in ("inputs", "configs", "src", "workflow"):
        shutil.copytree(
            ROOT / folder, tmp_path / folder, ignore=shutil.ignore_patterns("__pycache__")
        )
    for file in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(ROOT / file, tmp_path / file)
    return tmp_path


def test_config_cycle_and_invalid_panel(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml")
    (tmp_path / "b.yaml").write_text("extends: a.yaml")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


@pytest.mark.integration
def test_full_smoke_resume_tamper_detection_and_selection_only_change(tmp_path, config):
    root = isolated_project(tmp_path)
    runner = Runner(root, config)
    runner.run()
    base = root / "results/smoke"
    assert (base / "report/report.html").is_file()
    manifests = {
        stage: json.loads((base / stage / "manifest.json").read_text())
        for stage in ("prepare", "features", "fit", "predict", "select", "report")
    }
    runner.run()
    assert (
        json.loads((base / "fit/manifest.json").read_text())["started"]
        == manifests["fit"]["started"]
    )
    changed = deepcopy(config)
    changed["selection"]["seed"] += 1
    Runner(root, changed).run("select")
    assert (
        json.loads((base / "fit/manifest.json").read_text())["started"]
        == manifests["fit"]["started"]
    )
    assert (
        json.loads((base / "select/manifest.json").read_text())["signature"]
        != manifests["select"]["signature"]
    )
    (base / "prepare/core.csv").write_text("corrupted")
    assert not runner.current("prepare", runner.fingerprint("prepare"))
    with pytest.raises(ValueError, match="Damaged upstream"):
        runner.fingerprint("features")
