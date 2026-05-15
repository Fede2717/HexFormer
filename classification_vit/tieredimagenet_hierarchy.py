"""tieredImageNet fine -> superclass mapping.

Loaded dynamically at import time from a JSON file on the remote machine.
Falls back to a deterministic placeholder mapping so that import tests on
the local machine (where the dataset is not present) succeed.

JSON schema expected at ``_HIERARCHY_JSON_PATH``:
    {
        "num_fine":   608,
        "num_super":  34,
        "fine_to_super": {"0": 0, "1": 0, ..., "607": 33}
    }

The placeholder fallback constructs ``NUM_FINE`` fine classes distributed
evenly across ``NUM_SUPER`` superclasses. The fallback is only safe for
import-time checks; never run training on it.
"""
import json
import os
import warnings

_HIERARCHY_JSON_PATH = "/media/hdd/usr/forner/tieredImageNet/hierarchy.json"

# Canonical tieredImageNet sizes (Ren et al. 2018). Used both for the JSON
# schema validation and for the import-test fallback.
_DEFAULT_NUM_FINE = 608
_DEFAULT_NUM_SUPER = 34


def _load_from_json(path: str):
    with open(path, "r") as fh:
        data = json.load(fh)
    num_fine = int(data["num_fine"])
    num_super = int(data["num_super"])
    raw = data["fine_to_super"]
    fine_to_super = {int(k): int(v) for k, v in raw.items()}
    return fine_to_super, num_fine, num_super


def _placeholder_mapping(num_fine: int, num_super: int):
    """Even round-robin assignment so every super gets at least one fine."""
    return {f: f % num_super for f in range(num_fine)}


try:
    FINE_TO_SUPER, NUM_FINE, NUM_SUPER = _load_from_json(_HIERARCHY_JSON_PATH)
except (FileNotFoundError, OSError, KeyError, ValueError, json.JSONDecodeError) as _e:
    warnings.warn(
        f"tieredimagenet_hierarchy: could not load {_HIERARCHY_JSON_PATH} "
        f"({_e!r}); using placeholder mapping with "
        f"NUM_FINE={_DEFAULT_NUM_FINE}, NUM_SUPER={_DEFAULT_NUM_SUPER}. "
        f"This is only safe for import-time checks; do NOT train on it."
    )
    NUM_FINE = _DEFAULT_NUM_FINE
    NUM_SUPER = _DEFAULT_NUM_SUPER
    FINE_TO_SUPER = _placeholder_mapping(NUM_FINE, NUM_SUPER)


assert len(FINE_TO_SUPER) == NUM_FINE
assert set(FINE_TO_SUPER.values()).issubset(set(range(NUM_SUPER)))
