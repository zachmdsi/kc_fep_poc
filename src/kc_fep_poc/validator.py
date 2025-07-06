import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from .metrics import compute_metrics


def validate(csv_path: str | Path, base_dir: str | Path | None = None) -> bool:
    csv_path = Path(csv_path)
    if base_dir is None:
        base_dir = csv_path.parent
    base_dir = Path(base_dir)

    g_vals: list[float] = []
    f_vals: list[float] = []

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        if "file" not in reader.fieldnames:
            raise ValueError("CSV must contain 'file' column")

        for row in reader:
            file_path = base_dir / row["file"]
            obs = np.fromfile(file_path, dtype=np.uint8)
            m = compute_metrics(obs, float(obs.mean()))
            if m.k_hat > m.k_lzma:
                return False
            g_vals.append(m.g_t)
            f_vals.append(m.free_energy)

    if len(g_vals) < 2:
        return False

    r = float(np.corrcoef(g_vals, f_vals)[0, 1])
    if np.isnan(r) or r < 0.9:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate experiment logs")
    parser.add_argument("csv", help="metrics CSV file")
    parser.add_argument(
        "--base", help="base directory for observation files", default=None
    )
    args = parser.parse_args(argv)

    ok = validate(args.csv, args.base)
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
