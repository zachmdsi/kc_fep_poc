from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import generate_observations

DEFAULT_STEPS = 4096
DEFAULT_RUNS = 100
P_VALUES = [round(i / 10, 1) for i in range(1, 10)]


def write_dataset(
    output_dir: str | Path,
    steps: int = DEFAULT_STEPS,
    runs: int = DEFAULT_RUNS,
    seed: int | None = None,
) -> None:
    """Write binary observation files for a range of Bernoulli parameters.

    Parameters
    ----------
    output_dir : str or Path
        Directory in which to store generated `.bin` files. Subdirectories for
        each probability value are created automatically.
    steps : int, optional
        Number of time steps per sequence. Defaults to ``4096``.
    runs : int, optional
        Number of sequences per probability value. Defaults to ``100``.
    seed : int, optional
        Base random seed. If ``None``, each run is seeded with
        ``hash((p, i)) & 0xFFFF`` where ``p`` is the Bernoulli parameter and
        ``i`` is the run index.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metadata_file = out_path / "metadata.json"
    if metadata_file.exists():
        metadata: dict[str, dict[str, int | float]] = json.loads(
            metadata_file.read_text()
        )
    else:
        metadata = {}

    for p in P_VALUES:
        subdir = out_path / f"p{p}"
        subdir.mkdir(exist_ok=True)
        for i in range(runs):
            run_seed = seed if seed is not None else (hash((p, i)) & 0xFFFF)
            rng = np.random.default_rng(run_seed)
            obs = generate_observations(steps, p, rng)
            filename = subdir / f"run_{i:03d}.bin"
            obs.tofile(filename)
            key = str(filename.relative_to(out_path))
            metadata[key] = {"p": p, "seed": int(run_seed), "steps": steps}

    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate binary Bernoulli dataset")
    parser.add_argument(
        "output",
        nargs="?",
        default="dataset",
        help="directory to store generated files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="base random seed for reproducible data generation",
    )
    args = parser.parse_args(argv)
    write_dataset(args.output, seed=args.seed)


if __name__ == "__main__":
    main()
