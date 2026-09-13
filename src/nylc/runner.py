"""Portable stage execution with checksum-based restart and transactional publication."""

from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import json
import logging
import os
import platform
import sys
import traceback
import uuid
from threadpoolctl import threadpool_limits
from nylc.provenance import (
    sha256_file,
    write_json,
    digest_json,
    code_identity,
    software_identity,
    utc_now,
)
from nylc.runtime import setup_runtime
from nylc.stages import STAGES, DEPENDENCIES, CONFIG_KEYS


class Runner:
    def __init__(self, root, config):
        self.root = Path(root).resolve()
        self.config = config
        self.base = self.root / "results" / config["experiment"]
        self.code = code_identity(self.root)
        self.software = software_identity()
        setup_runtime(config)

    def stage_configuration(self, stage):
        config = {k: self.config[k] for k in CONFIG_KEYS[stage]}
        if stage in {"fit", "predict"}:
            config["sources"] = self.config["selection"]["sources"]
        if stage in {"diagnostics", "evaluate", "fit", "predict"}:
            config["threads"] = self.config["runtime"]["threads"]
        return config

    def fingerprint(self, stage):
        inputs = {}
        if stage == "prepare":
            inputs = {
                key: sha256_file(self.root / path) for key, path in self.config["inputs"].items()
            }
        if stage == "features":
            for name, settings in self.config["features"]["learned"].items():
                if name in self.config["features"]["sources"] and settings.get("checkpoint"):
                    inputs[name] = sha256_file(self.root / settings["checkpoint"])
        dependencies = {}
        for parent in DEPENDENCIES[stage]:
            path = self.base / parent / "manifest.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing {parent}; run dependencies first")
            manifest = json.loads(path.read_text())
            if manifest["signature"] != self.fingerprint(parent):
                raise ValueError(
                    f"Stale upstream stage: {parent}; run the pipeline to refresh dependencies"
                )
            for name, expected in manifest["outputs"].items():
                file = self.base / parent / name
                if not file.is_file() or sha256_file(file) != expected:
                    raise ValueError(f"Damaged upstream artifact: {parent}/{name}; rerun pipeline")
            dependencies[parent] = {
                "signature": manifest["signature"],
                "artifacts": {
                    name: checksum
                    for name, checksum in manifest["outputs"].items()
                    if name != "stage.log"
                },
            }
        return digest_json(
            {
                "stage": stage,
                "config": self.stage_configuration(stage),
                "inputs": inputs,
                "dependencies": dependencies,
                "code": self.code["sha256"],
                "software": {
                    "python": platform.python_version(),
                    "packages": self.software["packages"],
                    "os": sys.platform,
                    "machine": platform.machine(),
                },
            }
        )

    def current(self, stage, signature):
        path = self.base / stage / "manifest.json"
        if not path.exists():
            return False
        manifest = json.loads(path.read_text())
        return manifest.get("signature") == signature and all(
            (self.base / stage / name).is_file()
            and sha256_file(self.base / stage / name) == expected
            for name, expected in manifest["outputs"].items()
        )

    def run_stage(self, stage, force=False):
        signature = self.fingerprint(stage)
        if not force and self.current(stage, signature):
            logging.info("%s: current", stage)
            return False
        logging.info("%s: running", stage)
        stage_root = self.base / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        staging = self.root / ".work" / self.config["experiment"] / f"{stage}-{uuid.uuid4().hex}"
        staging.mkdir(parents=True)
        lock = stage_root / ".running"
        # Exclusive create prevents concurrent writers from mixing one experiment.
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\nstarted={utc_now()}\n")
        started = utc_now()
        try:
            with (staging / "stage.log").open("w", encoding="utf-8") as log:
                with (
                    redirect_stdout(log),
                    redirect_stderr(log),
                    threadpool_limits(limits=self.config["runtime"]["threads"]),
                ):
                    STAGES[stage](self.config, self.root, staging)
            outputs = {p.name: sha256_file(p) for p in staging.iterdir() if p.is_file()}
            previous_path = stage_root / "manifest.json"
            previous_outputs = (
                json.loads(previous_path.read_text()).get("outputs", {})
                if previous_path.exists()
                else {}
            )
            write_json(
                staging / "manifest.json",
                {
                    "stage": stage,
                    "status": "complete",
                    "signature": signature,
                    "started": started,
                    "finished": utc_now(),
                    "configuration": self.config,
                    "stage_configuration": self.stage_configuration(stage),
                    "code": self.code,
                    "software": self.software,
                    "outputs": outputs,
                },
            )
            for file in sorted(staging.iterdir(), key=lambda p: p.name == "manifest.json"):
                os.replace(file, stage_root / file.name)
            for name in set(previous_outputs) - set(outputs):
                if Path(name).name != name:
                    raise ValueError("Invalid previous artifact name")
                (stage_root / name).unlink(missing_ok=True)
            staging.rmdir()
            logging.info("%s: complete", stage)
            return True
        except BaseException:
            (staging / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            logging.error("%s failed; diagnostics: %s", stage, staging)
            raise
        finally:
            lock.unlink(missing_ok=True)

    def run(self, target="report", force=False):
        visited = set()

        def visit(stage):
            if stage in visited:
                return
            for dependency in DEPENDENCIES[stage]:
                visit(dependency)
            self.run_stage(stage, force=force)
            visited.add(stage)

        visit(target)
