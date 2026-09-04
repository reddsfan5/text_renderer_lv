"""Post-processing shared by dataset generation entry points."""

from pathlib import Path


def visualize_lmdb(lmdb_path: Path, sample_count: int) -> None:
    """Export LMDB crops, labels, and annotated previews to ``samples/``."""
    from lmdbs.lmdbs.visualize import visualize_full

    visualize_full(str(lmdb_path), num=sample_count)
