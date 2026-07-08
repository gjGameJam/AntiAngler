import argparse
from pathlib import Path

import geopandas as gpd
import geodatasets

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot filtered MPAs over a Natural Earth basemap."
    )
    p.add_argument(
        "--save", type=Path, default=None,
        help="Write PNG to this path instead of showing interactively (headless-safe).",
    )
    return p.parse_args()


args = parse_args()

# matplotlib.use() must run before pyplot is imported.
import matplotlib
if args.save:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

mpas = gpd.read_file(REPO_ROOT / "data" / "processed" / "wdpa_marine_ia_ib.gpkg")
world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

fig, ax = plt.subplots(figsize=(14, 8))
world.plot(ax=ax, color="lightgrey", edgecolor="white")
mpas.plot(ax=ax, color="cyan", edgecolor="black", alpha=0.7)
ax.set_title("Marine Protected Areas (Ia/Ib)")

if args.save:
    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.save}")
else:
    plt.show()
