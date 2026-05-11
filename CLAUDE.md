# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AntiAngler is a geospatial data processing project that filters the World Database of Protected Areas (WDPA) for marine protected areas (MPAs) with IUCN category Ia (Strict Nature Reserve) or Ib (Wilderness Area), then outputs them for analysis and visualization.

## Running Scripts

```bash
# Step 1: Filter raw WDPA shapefiles → processed GeoPackage
python scripts/prep_polygons.py

# Step 2: Visualize filtered MPAs on a world map
python scripts/view_polygons.py
```

There is no build system, test suite, or requirements file. Dependencies must be installed manually: `geopandas`, `pandas`, `geodatasets`, `matplotlib`.

## Data Layout

- `data/raw/WDPA/` — Raw WDPA September 2025 shapefiles, split into 3 parts (`WDPA_Sep2025_Public_shp_0`, `_1`, `_2`) due to size. These are gitignored.
- `data/processed/wdpa_marine_ia_ib.gpkg` — Output: filtered marine MPAs in WGS84 (EPSG:4326).
- `data/raw/WDPA/Resources_in_English/` — Official WDPA attribute/metadata PDFs for reference.

## Pipeline Architecture

`prep_polygons.py` reads each of the 3 split shapefiles, concatenates them into a single GeoDataFrame, filters rows where `MARINE` is `1` or `2` and `IUCN_CAT` is `Ia` or `Ib`, reprojects to EPSG:4326, and writes to a GeoPackage.

`view_polygons.py` loads the processed GeoPackage and overlays it on a Natural Earth world background via `geodatasets`.

The WDPA split-shapefile format is documented in `data/raw/WDPA/Shapefile_splitting_README.txt`.
