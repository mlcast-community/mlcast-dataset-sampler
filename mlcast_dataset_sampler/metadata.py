"""Read/write the sampling parameters carried inside a stats parquet file.

Parquet supports schema-level key/value metadata, so instead of encoding
the datacube shape, stride, date range, etc. in the output filename (and
re-parsing it in downstream steps) we stash it directly in the parquet
file under a single JSON-encoded key.

Downstream commands (`stats-heavy`, `sample`) read the metadata with
`read_stats_metadata(path)` and use it as the source of truth; the
filename stays human-readable but is no longer load-bearing.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import pyarrow.parquet as pq


STATS_METADATA_KEY = b"mlcast.stats"
SCHEMA_VERSION = 1


def encode_stats_metadata(params: Mapping[str, Any]) -> dict[bytes, bytes]:
    """Build the schema-level metadata dict for a stats parquet file."""
    payload = dict(params)
    payload["schema_version"] = SCHEMA_VERSION
    return {STATS_METADATA_KEY: json.dumps(payload, sort_keys=True).encode()}


def read_stats_metadata(path: str) -> dict[str, Any]:
    """Load the mlcast sampling parameters from a stats parquet file.

    Raises
    ------
    KeyError
        If the file has no `mlcast.stats` metadata key — e.g. it was not
        produced by the mlcast sampler or is from an older version.
    """
    schema = pq.read_schema(path)
    if schema.metadata is None or STATS_METADATA_KEY not in schema.metadata:
        raise KeyError(
            f"{path}: no 'mlcast.stats' metadata found in parquet schema. "
            f"Not a stats file produced by mlcast-dataset-sampler?"
        )
    return json.loads(schema.metadata[STATS_METADATA_KEY].decode())
