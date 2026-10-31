"""Data-source loaders (one module per domain); importing this package registers all of them in SOURCES."""
from .base import SOURCES, Source  # noqa: F401
from . import satellite, context, inventory, pollution, embeddings  # noqa: E402,F401
