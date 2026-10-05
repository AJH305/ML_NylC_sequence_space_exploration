"""Copy the figures stage output into the LaTeX tree.

The pipeline deliberately writes only inside ``results/<experiment>/``. This
script is the one explicit step that publishes those files to the thesis, so a
pipeline run never modifies the LaTeX sources as a side effect.

    python tools/sync_thesis_figures.py
    python tools/sync_thesis_figures.py --experiment reference_cpu --formats pdf png

Include the copied files without scaling, otherwise their font sizes no longer
match the body text::

    \\includegraphics[width=\\textwidth]{figures/<name>.pdf}
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="ML_NylC project directory")
    parser.add_argument("--experiment", default="reference_cpu")
    parser.add_argument(
        "--destination",
        default="../Thesis/figures",
        help="target directory, relative to --root",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf"],
        choices=["pdf", "png"],
        help="PDF is what LaTeX includes; PNG is only a preview",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    source = root / "results" / args.experiment / "figures"
    destination = (root / args.destination).resolve()
    if not source.is_dir():
        parser.error(f"No figures stage output at {source}; run `nylc stage figures` first")

    style = source / "style.json"
    if style.is_file():
        recorded = json.loads(style.read_text(encoding="utf-8"))
        if not recorded.get("font_matches_thesis", False):
            print(
                f"WARNING: figures were rendered with {recorded.get('font_resolved')!r}, "
                "which is not Arial or Liberation Sans; typography will not match "
                "the thesis.",
                file=sys.stderr,
            )
        for name, missing in (recorded.get("figures_skipped") or {}).items():
            print(f"note: {name} was not built (missing {', '.join(missing)})", file=sys.stderr)

    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for suffix in args.formats:
        for path in sorted(source.glob(f"*.{suffix}")):
            shutil.copy2(path, destination / path.name)
            copied += 1
    print(f"Copied {copied} file(s) from {source} to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
