"""Bundle ablation/EDA/reproduction evidence without copying cutoff caches."""

import argparse
import zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--include-checkpoints", action="store_true")
    p.add_argument("--predictions", type=Path, nargs="*", default=[])
    a = p.parse_args()
    if not a.run.is_dir():
        raise FileNotFoundError(a.run)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, "x", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(a.run.rglob("*")):
            if not path.is_file() or path.suffix in (".npz", ".npy"):
                continue
            if path.suffix == ".pt" and not a.include_checkpoints:
                continue
            z.write(path, Path("run") / path.relative_to(a.run))
        for path in a.predictions:
            z.write(path, Path("submission") / path.name)
        for folder in ("configs", "docs"):
            for path in sorted(Path(folder).rglob("*")):
                if path.is_file():
                    z.write(path, path)
    print(a.out)


if __name__ == "__main__":
    main()
