"""
Count the input files and events a data config provides.

Counting reads only file metadata (ROOT tree entries / HDF5 dataset shapes),
the same pass the loaders already make when they are constructed, so it does
not read any particle data.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .base_loader import ConfigType
from .dataset import create_loader

logger = logging.getLogger(__name__)


@dataclass
class FileEventCount:
    """Events contributed by one input file.

    Attributes:
        path: Resolved file path (globs expanded).
        n_events: Number of events in the file.
        label: Event label from ``file_event_label_map`` (-1 if unmatched).
    """

    path: Path
    n_events: int
    label: int = -1


@dataclass
class EventCount:
    """Input files and total events available from a data config.

    Attributes:
        total: Total number of events across all input files, before
            ``max_events`` is applied.
        per_file: Per-file breakdown, in loader order.
    """

    total: int
    per_file: list[FileEventCount] = field(default_factory=list)

    @property
    def n_files(self) -> int:
        """Number of input files that contribute events."""
        return len(self.per_file)

    def n_used(self, max_events: int | None = None) -> int:
        """Return how many events training uses given a ``max_events`` cap.

        Args:
            max_events: Cap on the number of events (None or 0 means no cap,
                matching ``create_datasets``).

        Returns:
            ``min(total, max_events)`` if capped, else ``total``.
        """
        if max_events:
            return min(self.total, max_events)
        return self.total


def count_events(config: ConfigType) -> EventCount:
    """
    Count the input files and events a data config provides.

    Args:
        config: Data configuration (Hydra DictConfig or plain dict), i.e.
            ``cfg.data``.

    Returns:
        EventCount with the total and per-file breakdown.
    """
    loader = create_loader(config)
    try:
        labels: list[int] = getattr(loader, "_event_labels", [])
        per_file = [
            FileEventCount(
                path=path,
                n_events=n,
                label=labels[i] if i < len(labels) else -1,
            )
            for i, (path, n) in enumerate(loader.file_event_counts())
        ]
        return EventCount(total=len(loader), per_file=per_file)
    finally:
        close = getattr(loader, "close", None)
        if callable(close):
            close()
