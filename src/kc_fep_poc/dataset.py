from __future__ import annotations

from pathlib import Path
import argparse

from .metrics import generate_observations


DEFAULT_STEPS = 4096
DEFAULT_RUNS = 100
P_VALUES = [round(i / 10, 1) for i in range(1, 10)]


def write_dataset(output_dir: str | Path,
                  steps: int = DEFAULT_STEPS,
                  runs: int = DEFAULT_RUNS) -> None:
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
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for p in P_VALUES:
        subdir = out_path / f"p{p}"
        subdir.mkdir(exist_ok=True)
        for i in range(runs):
            obs = generate_observations(steps, p)
            filename = subdir / f"run_{i:03d}.bin"
            filename.write_bytes(bytes(obs))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate binary Bernoulli dataset")
    parser.add_argument("output", nargs="?", default="dataset",
                        help="directory to store generated files")
    args = parser.parse_args(argv)
    write_dataset(args.output)


if __name__ == "__main__":
    main()
