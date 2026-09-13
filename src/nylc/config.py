"""Validated experiment configuration; paths are anchored to the project."""

from copy import deepcopy
from pathlib import Path
import re
import json
import yaml
from jsonschema import Draft202012Validator, ValidationError


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def read_config(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError("Circular configuration inheritance")
    seen.add(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    parent = config.pop("extends", None)
    if parent:
        config = merge(read_config(path.parent / parent, seen), config)
    return config


def load_config(path):
    config = read_config(path)
    schema = json.loads(Path(__file__).with_name("config.schema.json").read_text(encoding="utf-8"))
    try:
        Draft202012Validator(schema).validate(config)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid configuration at {'.'.join(map(str, exc.path))}: {exc.message}"
        ) from exc
    required = {
        "schema_version",
        "experiment",
        "inputs",
        "data",
        "features",
        "gp",
        "validation",
        "baseline_comparison",
        "selection",
        "runtime",
    }
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError(f"Expected configuration keys {sorted(required)} and schema_version 1")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["experiment"]):
        raise ValueError("Experiment must be a simple directory name")
    if config["runtime"]["device"] not in {"cpu", "cuda", "auto"}:
        raise ValueError("Device must be cpu, cuda or auto")
    for key in ("threads", "batch_size"):
        if not isinstance(config["runtime"][key], int) or config["runtime"][key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("inner_restarts", "final_restarts", "max_iterations"):
        if not isinstance(config["gp"][key], int) or config["gp"][key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for low, high in config["gp"]["bounds"].values():
        if not 0 < low < high:
            raise ValueError("Optimizer bounds must be positive and ordered")
    selection = config["selection"]
    k = selection["panel_size"] - selection["controls"]
    if selection["controls"] < 0 or k < 1:
        raise ValueError("Panel must contain at least one generated candidate")
    if not selection["shortlist_sizes"] or min(selection["shortlist_sizes"]) < k:
        raise ValueError("All shortlists must accommodate the generated panel")
    if selection["primary_shortlist"] not in selection["shortlist_sizes"]:
        raise ValueError("Primary shortlist must be present in shortlist_sizes")
    if not 0 < selection["activity_fraction"] <= 1 or selection["sensitivity_seeds"] < 1:
        raise ValueError("Invalid activity fraction or sensitivity seed count")
    if not set(selection["sources"]).issubset(config["features"]["sources"]):
        raise ValueError("Selection sources must be included in features.sources")
    from nylc.features.descriptors import DESCRIPTOR_SOURCES

    if not config["features"]["sources"]:
        raise ValueError("At least one feature source is required")
    for name in config["features"]["sources"]:
        if name not in DESCRIPTOR_SOURCES and name not in config["features"]["learned"]:
            raise ValueError(f"Missing learned model configuration: {name}")
    return config
