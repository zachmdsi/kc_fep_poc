import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kc_fep_poc.dataset import write_dataset  # noqa: E402


def test_write_dataset(tmp_path: Path):
    write_dataset(tmp_path)
    dirs = list(tmp_path.iterdir())
    assert len(dirs) == 9  # p=0.1..0.9
    total_files = sum(1 for d in dirs for _ in d.iterdir())
    assert total_files == 900
    # check a single file length
    sample_file = next(dirs[0].iterdir())
    assert sample_file.stat().st_size == 4096


def test_write_dataset_seed_reproducible(tmp_path: Path):
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    write_dataset(d1, steps=16, runs=1, seed=123)
    write_dataset(d2, steps=16, runs=1, seed=123)
    f1 = d1 / "p0.1" / "run_000.bin"
    f2 = d2 / "p0.1" / "run_000.bin"
    assert f1.read_bytes() == f2.read_bytes()
