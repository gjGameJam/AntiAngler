import geopandas as gpd
import geodatasets
import matplotlib.pyplot as plt

# Load the filtered polygons (or use mpas from your script)
mpas = gpd.read_file("../data/processed/wdpa_marine_ia_ib.gpkg")

# Plot the polygons
world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
fig, ax = plt.subplots(figsize=(14,8))
world.plot(ax=ax, color="lightgrey", edgecolor="white")
mpas.plot(ax=ax, color="cyan", edgecolor="black", alpha=0.7)
ax.set_title("Marine Protected Areas (Ia/Ib)")
plt.show()
