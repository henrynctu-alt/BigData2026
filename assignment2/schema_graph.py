"""Draw all official table PKs and every FK, including self references."""

import argparse
import importlib.util
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("reports/schema"))
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location(
        "preprocess", "official/scripts/preprocess.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lines = [
        "digraph schema {",
        "rankdir=LR; nodesep=0.6; ranksep=1; splines=polyline;",
        'node [shape=plain,fontsize=11,fontname="Helvetica"];',
        'edge [color="#64748b",arrowsize=0.65];',
    ]
    for name, table in module.TABLES.items():
        cells = [("PK", table["pk"])] + [("FK", key) for key in table["fks"]]
        label = '<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">'
        label += f'<TR><TD BGCOLOR="#dceafa"><B>{name}</B></TD></TR>'
        for kind, key in cells:
            label += f'<TR><TD PORT="{key}" ALIGN="LEFT">{kind}: {key}</TD></TR>'
        label += "</TABLE>"
        lines.append(f"{name} [label=<{label}>];")
    for name, table in module.TABLES.items():
        for key, target in table["fks"].items():
            dest, pk = (
                target
                if isinstance(target, (tuple, list))
                else (target, module.TABLES[target]["pk"])
            )
            lines.append(f"{name}:{key}:e -> {dest}:{pk}:w;")
    lines.append("}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".dot").write_text("\n".join(lines))
    for ext in ("svg", "pdf"):
        subprocess.run(
            [
                "dot",
                f"-T{ext}",
                str(a.out.with_suffix(".dot")),
                "-o",
                str(a.out.with_suffix("." + ext)),
            ],
            check=True,
        )
    print(f"Wrote {a.out}.dot/.svg/.pdf")


if __name__ == "__main__":
    main()
