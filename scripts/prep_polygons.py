import geopandas as gpd
from pathlib import Path
import pandas as pd

# The objective of this script:
# - Read from all three WDPA shapefile parts (0, 1, 2)
# - Merge them into one GeoDataFrame
# - Filter for marine protected areas in category Ia or Ib
# - Save to a single GeoPackage for downstream use

# Paths to shapefiles (3 parts)
shp_paths = [
    Path("../data/raw/WDPA/WDPA_Sep2025_Public_shp_0/WDPA_Sep2025_Public_shp-polygons.shp"),
    Path("../data/raw/WDPA/WDPA_Sep2025_Public_shp_1/WDPA_Sep2025_Public_shp-polygons.shp"),
    Path("../data/raw/WDPA/WDPA_Sep2025_Public_shp_2/WDPA_Sep2025_Public_shp-polygons.shp"),
]
#print(shp_paths[0].exists())
for p in shp_paths:
    print(f"Reading {p}")
    gdf = gpd.read_file(p, on_invalid="ignore")
    print(f"{p}: {len(gdf)} features, CRS={gdf.crs}")


# Read and merge
frames = [gpd.read_file(p) for p in shp_paths]
wdpa = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
print(f"Total features after merge: {len(wdpa)}")

# Inspect first few rows
#print("Sample rows:")
#print(wdpa[["NAME", "MARINE", "IUCN_CAT"]].head(10))

# Clean columns
wdpa["MARINE"] = pd.to_numeric(wdpa["MARINE"], errors='coerce')
wdpa["IUCN_CAT"] = wdpa["IUCN_CAT"].astype(str).str.strip()

#only include marine areas of Ia or Ib classification
mpas = wdpa[
    (wdpa["MARINE"].isin([1, 2])) &   # include fully marine or mixed
    (wdpa["IUCN_CAT"].isin(["Ia", "Ib"]))
]

print(f"Filtered MPAs: {len(mpas)}")

# Normalize CRS to WGS84 (lat/lon)
mpas = mpas.to_crs("EPSG:4326")

print(f"Original WDPA: {len(wdpa)} features")
print(f"Filtered MPAs (Ia/Ib): {len(mpas)} features")

# Save processed polygons
out_file = Path("../data/processed/wdpa_marine_ia_ib.gpkg")
mpas.to_file(out_file, driver="GPKG")
print(f"Saved filtered polygons → {out_file}")
