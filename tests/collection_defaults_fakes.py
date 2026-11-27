"""Patch a modified copy of config/collection_defaults.yaml in for one test."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from umi.config import collection_defaults
from umi.infra.file_store import read_yaml
from umi.infra.file_store import write_yaml


def defaults_document() -> dict:
    return deepcopy(read_yaml(collection_defaults.DEFAULT_PATH))


def patched_defaults(directory: Path, document: dict):
    path = Path(directory) / 'collection_defaults.yaml'
    write_yaml(path, document)
    return patch.object(collection_defaults, 'current', return_value=collection_defaults.load(path))
