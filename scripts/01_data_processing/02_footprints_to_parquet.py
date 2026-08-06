import gc
import os
import sys
import time
import atexit
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.stats import genpareto
from dotenv import load_dotenv

#Load environment variables
load_dotenv()

ROOT_DIR = Path(os.environ["ROOT_DIR"])
HOT_DRY_EVENTS_DIR = Path(os.environ["HOT_DRY_EVENTS_DIR"])
FOOTPRINTS_PARQUET_DIR = Path(os.environ["PARQUETDIR"])
LAND_SEA_MASK_PATH = Path(os.environ["SFTLF_PATH"])


MASK_OCEAN = os.environ.get("MASK_OCEAN", "False").lower() == "true"
DATA_USED = os.environ.get("DATA_USED", "canari").lower()
VERSION = os.environ.get("VERSION", "0_1_3")

# Get all files startwith event_footprints
event_footprints_files = sorted(str(p) for p in Path(HOT_DRY_EVENTS_DIR).rglob(f"event_footprints_{DATA_USED}_*.nc"))

# Group all event footprints into a single xarray dataset
event_footprints_ds = xr.Dataset()
datasets = []
event_id_offset = 0

for f in event_footprints_files:
    ds = xr.open_dataset(f)

    n_events = ds.sizes["event_id"]
    if DATA_USED != "era5_zarr":
        member_id = int(Path(f).stem.split("_")[4])
    else:
        member_id = int(Path(f).stem.split("_")[3])

    # make event_id unique across files 
    ds = ds.assign_coords(
        event_id=np.arange(event_id_offset + 1, event_id_offset + n_events + 1),
        member=("event_id", np.full(n_events, member_id, dtype=int)),
    )

    datasets.append(ds)
    event_id_offset += n_events

event_footprints_ds = xr.concat(
    datasets,
    dim="event_id"
)

# Select only JJA events
month = event_footprints_ds["event_start"].dt.month

event_footprints_ds_jja = event_footprints_ds.where(
    (month >= 6) & (month <= 8),
    drop=True
)

if MASK_OCEAN:

    num_event_days = event_footprints_ds_jja["event_footprints"].sel(
        field="num_event_days",
        drop=True,
    )

    sftlf = xr.open_dataset(LAND_SEA_MASK_PATH)
    sftlf.drop_vars("type")
    sftlf = sftlf['sftlf']  # select the variable from the dataset
    sftlf = sftlf.rename({"lon": "lon_um_atmos_grid_t", "lat": "lat_um_atmos_grid_t"})
    sftlf = sftlf.reindex_like(ds, method="nearest", tolerance=0.1)
    # Shift longitudes from [0, 360] to [-180, 180]
    sftlf_lon = sftlf["lon_um_atmos_grid_t"].values
    new_lon = (((sftlf_lon + 180) % 360) - 180).astype(sftlf_lon.dtype)

    # Assign new longitudes and sort
    dims = sftlf["lon_um_atmos_grid_t"].dims
    sftlf = sftlf.assign_coords({"lon_um_atmos_grid_t": (dims, new_lon)})
    sftlf = sftlf.sortby("lon_um_atmos_grid_t")
    sftlf = sftlf.sel(lon_um_atmos_grid_t=slice(-12, 42), lat_um_atmos_grid_t=slice(35.5, 71))

    # Longitude conversion can create a duplicate at 0/360, so remove duplicates.
    sftlf = sftlf.sortby("lon_um_atmos_grid_t")
    sftlf = sftlf.isel(
        lon_um_atmos_grid_t=~sftlf.get_index("lon_um_atmos_grid_t").duplicated(),
        lat_um_atmos_grid_t=~sftlf.get_index("lat_um_atmos_grid_t").duplicated(),
    )

    # Rename coordinates to match the event footprints dataset
    sftlf = sftlf.rename({
        "lon_um_atmos_grid_t": "lon",
        "lat_um_atmos_grid_t": "lat",
    })

    # Put the land-sea mask exactly onto the event grid.
    sftlf_at_events = sftlf.interp(
        lat=num_event_days["lat"],
        lon=num_event_days["lon"],
        method="nearest",
    )

    on_land = sftlf_at_events > 30
    above_threshold = num_event_days > 0.3

    land_pixels = (on_land & above_threshold).sum(dim=("lat", "lon"))
    event_pixels = above_threshold.sum(dim=("lat", "lon"))

    # Avoid division by zero for events with no pixels above the threshold.
    land_proportion = xr.where(
        event_pixels > 0,
        land_pixels / event_pixels,
        0.0,
    )

    event_footprints_ds_jja = event_footprints_ds_jja.where(
        land_proportion > 0.5,
        drop=True,
    )


# event_footprints: DataArray with dims (event_id, field, lat, lon)
da = event_footprints_ds_jja["event_footprints"].rename("value")

# Convert to long dataframe
df = da.to_dataframe().reset_index()

# Clean field names
df["field"] = df["field"].replace({
    "max_-spi": "max_neg_spi"
})

# Pivot to one row per event_id, lat, lon
df_wide = (
    df.pivot_table(
        index=["event_id", "lat", "lon"],
        columns="field",
        values="value"
    )
    .reset_index()
)


# Build event-level metadata table from event_id coordinates
# Keep CFTime as string
meta = pd.DataFrame({
    "event_id": da["event_id"].values,
    "event_start": [str(t) for t in da["event_start"].values],
    "event_end": [str(t) for t in da["event_end"].values],
    "event_rp": da["return_period_years"].values,
    "max_event_temp": da["max_event_temp"].values,
    "max_event_minus_spi": da["max_event_minus_spi"].values,
    "event_duration": da["event_duration"].values,
})

# Merge metadata onto every event_id × lat × lon row
df_out = df_wide.merge(meta, on="event_id", how="left")

# Add a grid column for unique lat-lon pairs
latlon = (
    df_out[["lat", "lon"]]
    .drop_duplicates()
    .sort_values(["lat", "lon"])
    .reset_index(drop=True)
)
latlon["grid"] = range(1, len(latlon) + 1)

df_out = df_out.merge(latlon, on=["lat", "lon"], how="left")

# Reorder columns
front_cols = [
    "grid", "event_id", "lat", "lon",
    "event_start", "event_end", "event_rp", "event_duration",
    "max_event_temp", "max_event_minus_spi",
    "max_temp", "max_neg_spi", "num_event_days",
]
remaining = [c for c in df_out.columns if c not in front_cols]
df_out = df_out[front_cols + remaining]

# Save parquet
df_out.to_parquet(f"{FOOTPRINTS_PARQUET_DIR}/event_footprints_long_jja_{VERSION}.parquet", index=False)

print(df_out.head())
print(df_out.shape)