"""Project defaults for the fields a batch collection.yaml may omit.

config/collection_defaults.yaml mirrors the collection.yaml layout, is tracked in Git and is
shared by every machine. There is deliberately no per-machine layer: a batch that needs a
different policy states it in its own collection.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from umi.infra.file_store import read_yaml
from umi.infra.paths import CONFIG_ROOT

DEFAULT_PATH = CONFIG_ROOT / 'collection_defaults.yaml'


@dataclass(frozen=True)
class CollectionDefaults:
    path: Path
    values: MappingProxyType

    def report(self) -> dict:
        return dict(path=str(self.path), values=dict(self.values))


def load(path: Path | None = None) -> CollectionDefaults:
    from umi.config.collection_schema import validate_collection_defaults
    path = (path or DEFAULT_PATH).expanduser().resolve()
    return CollectionDefaults(path, MappingProxyType(validate_collection_defaults(read_yaml(path))))


@lru_cache(maxsize=None)
def current() -> CollectionDefaults:
    return load()


def audio_sync_enabled() -> bool:
    return current().values['audio_sync']['enabled']


def audio_sync_algorithm() -> str:
    return current().values['audio_sync']['algorithm']


def poster_vio_base_tag_enabled() -> bool:
    return current().values['backend']['base_tag_enabled']


def auto_order() -> tuple[str, ...]:
    return current().values['backend']['auto_order']
