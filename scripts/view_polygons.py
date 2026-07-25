import argparse
from pathlib import Path

import geopandas as gpd
import geodatasets

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_GPKG = REPO_ROOT / "data" / "processed" / "wdpa_marine_ia_ib.gpkg"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Plot filtered MPAs over a Natural Earth basemap."
    )
    p.add_argument(
        "--gpkg", type=Path, default=DEFAULT_GPKG,
        help=f"Filtered-MPA GeoPackage to plot (default {DEFAULT_GPKG}).",
    )
    p.add_argument(
        "--save", type=Path, default=None,
        help="Write PNG to this path instead of showing interactively (headless-safe).",
    )
    return p.parse_args(argv)


def save_figure_atomic(fig, save_path):
    """Save the figure atomically: render to a sibling .tmp then rename over the
    target, so an interrupted render never leaves a truncated PNG. Cleaned up on
    failure."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = save_path.with_name(save_path.name + ".tmp")
    try:
        fig.savefig(tmp, dpi=150, bbox_inches="tight")
        tmp.replace(save_path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return save_path


def main(argv=None):
    args = parse_args(argv)

    if not args.gpkg.exists():
        raise SystemExit(
            f"MPA GeoPackage not found: {args.gpkg}\n"
            "Run scripts/prep_polygons.py first to build it from the WDPA shapefiles."
        )

    # matplotlib.use() must run before pyplot is imported, and the backend
    # depends on whether we are rendering headless (--save) or interactively.
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mpas = gpd.read_file(args.gpkg)
    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

    fig, ax = plt.subplots(figsize=(14, 8))
    world.plot(ax=ax, color="lightgrey", edgecolor="white")
    mpas.plot(ax=ax, color="cyan", edgecolor="black", alpha=0.7)
    ax.set_title("Marine Protected Areas (Ia/Ib)")

    if args.save:
        out = save_figure_atomic(fig, args.save)
        print(f"Wrote {out}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
