"""Data-source loaders (one module per domain); importing this package registers all of them in SOURCES."""
from .base import SOURCES, Source  # noqa: F401
from . import satellite, context, inventory, pollution, embeddings, daily, placebo, composite  # noqa: E402,F401
