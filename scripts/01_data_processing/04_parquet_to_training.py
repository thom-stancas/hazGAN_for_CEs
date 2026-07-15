import os
import warnings
from pathlib import Path

import scipy.stats
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cartopy
import glob

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

ROOT_DIR = Path(os.environ["ROOT_DIR"])
FOOTPRINTS_PARQUET_DIR = Path(os.environ["PARQUETDIR"])
HOT_DRY_EVENTS_DIR = Path(os.environ["HOT_DRY_EVENTS_DIR"])
OUTPUT_DIR = Path(os.environ["TRAINING_DIR"])


# ----------------------------------------------
# Plot function
# ----------------------------------------------

def plot_gpd_fits(raw_extra, var_list=None, save_dir=None):
    """Plot GPD parameter maps and sample densities for fields in var_list.

    raw_extra: xarray Dataset containing fields like pk_<var>, scale_<var>, shape_<var>, thresh_<var>
    var_list: list of variable names to plot. Defaults to the three FIELDS.
    save_dir: if provided, save figures to this directory with names f'fit_<var>.png'.
    """
    if var_list is None:
        var_list = ["max_temp", "max_neg_spi", "num_event_days"]

    for var in var_list:
        p_crit = 0.05

        fig, axs = plt.subplots(1, 4, figsize=(16, 3), sharex=True, sharey=True,
                                subplot_kw={"projection": ccrs.PlateCarree()})
        ax4 = fig.add_axes([0.825, 0.1, 0.15, 0.8])
        plt.tight_layout()
        plt.subplots_adjust(right=0.8)
        cmap = "PuBu_r"

        p_cmap = plt.get_cmap(cmap).copy()
        p_cmap.set_under("crimson")

        pk_da = raw_extra.sel(field=f"pk_{var}").mean("event_id")
        scale_da = raw_extra.sel(field=f"scale_{var}").mean("event_id")
        shape_da = raw_extra.sel(field=f"shape_{var}").mean("event_id")
        thresh_da = raw_extra.sel(field=f"thresh_{var}").mean("event_id")

        pk_da.plot(ax=axs[0], cmap=p_cmap, vmin=p_crit, cbar_kwargs={"label": None})
        thresh_da.plot(ax=axs[1], cmap=cmap, cbar_kwargs={"label": None})
        scale_da.plot(ax=axs[2], cmap=cmap, cbar_kwargs={"label": None})
        shape_da.plot(ax=axs[3], cmap=cmap, add_colorbar=False)

        dist = scipy.stats.genpareto
        shapes_all = raw_extra.sel(field=f'shape_{var}').values
        percentiles = np.linspace(0.01, 0.99, 10)
        shapes = raw_extra.sel(field=f'shape_{var}').quantile(percentiles)
        loc = raw_extra.sel(field=f'thresh_{var}').mean()
        scale = raw_extra.sel(field=f'scale_{var}').mean()
        vmin = float(np.nanmin(shapes_all))
        vmax = float(np.nanmax(shapes_all))
        norm = plt.Normalize(vmin, vmax)
        colors = [plt.get_cmap(cmap)(norm(value)) for value in shapes]

        for i, shape in enumerate(shapes):
            u = np.linspace(0.95, 0.999, 100)
            x = dist.ppf(u, shape)
            y = dist.pdf(x, shape)
            ax4.plot(x, y, label=f"ξ={shape:.2f}", color=colors[i])

        def percentage_formatter(x, pos):
            return f'{100 * x:.0f}%'

        ax4.set_xlabel("")
        ax4.set_ylabel("")
        ax4.yaxis.set_major_formatter(percentage_formatter)
        ax4.tick_params(direction='in')
        ax4.yaxis.set_label_position("right")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax4)
        cbar.set_label(None)
        cbar.set_label("ξ")
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.set_ylabel('ξ', rotation=0, labelpad=15)

        axs[0].set_title("H₀: X~GPD(ξ,μ,σ)")
        axs[1].set_title("μ")
        axs[2].set_title("σ")
        axs[3].set_title("ξ")

        for ax in axs[:-1].ravel():
            ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        if save_dir is not None:
            out = Path(save_dir) / f"fit_{var}.png"
            fig.savefig(out, dpi=300)
        plt.close(fig)




# ----------------------------------------------
# ----------------------------------------------
# Main code
# ----------------------------------------------
# ----------------------------------------------


def main():

    # Load the parquet files
    events_df = pd.read_parquet(FOOTPRINTS_PARQUET_DIR / "events_jja_0_1_2.parquet")
    event_long = pd.read_parquet(FOOTPRINTS_PARQUET_DIR / "event_footprints_long_jja_0_1_2.parquet")


    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    # Plot the GPD fits for the fields
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------


    # raw transformed fields back to event_id x lat x lon
    idx = events_df.set_index(["event_id", "lat", "lon"])

    raw_extra = (
        idx[
            [
                "shape.max_temp", "shape.max_neg_spi", "shape.num_event_days",
                "scale.max_temp", "scale.max_neg_spi", "scale.num_event_days",
                "pk.max_temp", "pk.max_neg_spi", "pk.num_event_days",
                "thresh.max_temp", "thresh.max_neg_spi", "thresh.num_event_days",
            ]
        ]
        .rename(columns=lambda c: c.replace(".", "_"))
        .astype("float64")
        .to_xarray()
        .to_array("field")
    )

    # Plot the GPD fits for the fields
    os.makedirs(FOOTPRINTS_PARQUET_DIR / "plots", exist_ok=True)

    plot_gpd_fits(raw_extra, var_list = ["max_temp", "max_neg_spi", "num_event_days"], 
                   save_dir=f"{FOOTPRINTS_PARQUET_DIR}/plots")


    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    # Check the columns in the DataFrame and prepare for training
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------

    # Check the columns in the DataFrame
    assert "event_id" in events_df.columns, "event_id column is missing"
    assert "grid" in events_df.columns, "grid column is missing"

    assert "max_temp" in events_df.columns, "max_temp column is missing"
    assert "max_neg_spi" in events_df.columns, "max_neg_spi column is missing"
    assert "num_event_days" in events_df.columns, "num_event_days column is missing"

    assert "time.max_temp" in events_df.columns, "time_max_temp column is missing"
    assert "time.max_neg_spi" in events_df.columns, "time_max_neg_spi column is missing"
    assert "time.num_event_days" in events_df.columns, "time_num_event_days column is missing"

    assert "ecdf.max_temp" in events_df.columns, "ecdf_max_temp column is missing"
    assert "ecdf.max_neg_spi" in events_df.columns, "ecdf_max_neg_spi column is missing"
    assert "ecdf.num_event_days" in events_df.columns, "ecdf_num_event_days column is missing"

    assert "scdf.max_temp" in events_df.columns, "scdf_max_temp column is missing"
    assert "scdf.max_neg_spi" in events_df.columns, "scdf_max_neg_spi column is missing"
    assert "scdf.num_event_days" in events_df.columns, "scdf_num_event_days column is missing"
        
    assert "event.rp" in events_df.columns, "event.rp column is missing"

    assert "thresh.max_temp" in events_df.columns, "thresh_max_temp column is missing"
    assert "thresh.max_neg_spi" in events_df.columns, "thresh_max_neg_spi column is missing"
    assert "thresh.num_event_days" in events_df.columns, "thresh_num_event_days column is missing"

    assert "shape.max_temp" in events_df.columns, "shape_max_temp column is missing"
    assert "shape.max_neg_spi" in events_df.columns, "shape_max_neg_spi column is missing"
    assert "shape.num_event_days" in events_df.columns, "shape_num_event_days column is missing"

    assert "scale.max_temp" in events_df.columns, "scale_max_temp column is missing"
    assert "scale.max_neg_spi" in events_df.columns, "scale_max_neg_spi column is missing"
    assert "scale.num_event_days" in events_df.columns, "scale_num_event_days column is missing"

    assert "pk.max_temp" in events_df.columns, "pk_max_temp column is missing"
    assert "pk.max_neg_spi" in events_df.columns, "pk_max_neg_spi column is missing"
    assert "pk.num_event_days" in events_df.columns, "pk_num_event_days column is missing"


    # Rename columns to replace "." with "_"
    events_df.columns = [c.replace(".", "_") for c in events_df.columns]

    # Add event_duration, event_start, and event_end to events_df
    event_duration = event_long[["event_id","grid","event_duration"]]
    event_start = event_long[["event_id","grid","event_start"]]
    event_end = event_long[["event_id","grid","event_end"]]
    events_df = events_df.merge(event_duration, on=["event_id", "grid"], how="left")
    events_df = events_df.merge(event_start, on=["event_id", "grid"], how="left")
    events_df = events_df.merge(event_end, on=["event_id", "grid"], how="left")


    # Convert time columns to datetime and calculate day_of_event
    events_df["event_start"] = pd.to_datetime(events_df["event_start"], utc=True, errors="coerce")
    events_df["event_end"] = pd.to_datetime(events_df["event_end"], utc=True, errors="coerce")
    events_df["time_max_temp"] = pd.to_datetime(events_df["time_max_temp"], utc=True, errors="coerce")
    events_df["day_of_event"] = (events_df["time_max_temp"] - events_df["event_start"]).dt.days.astype("float32")

    # Check ecdf ranges
    assert events_df["ecdf_max_temp"].max() <= 1
    assert events_df["ecdf_max_temp"].min() >= 0

    assert events_df["ecdf_max_neg_spi"].max() <= 1
    assert events_df["ecdf_max_neg_spi"].min() >= 0

    assert events_df["ecdf_num_event_days"].max() <= 1
    assert events_df["ecdf_num_event_days"].min() >= 0



    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    # Format the data for training
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------


    # Format the data for training
    FIELDS = ["max_temp", "max_neg_spi", "num_event_days"]

    # Number of unique events
    num_events = events_df["event_id"].nunique()

    events_df = events_df.sort_values(["event_id", "lat", "lon"])

    # Get the lat/lon grid from a hot-dry events netcdf file
    lat_lon_data = xr.open_dataset(f"{HOT_DRY_EVENTS_DIR}/hot_dry_labels_canari_1_97_-1.5_150_12.nc")
    lat = lat_lon_data.lat.values
    lon = lat_lon_data.lon.values
    ny, nx = len(lat), len(lon)
    grid = np.array(range(1, ny * nx + 1)).reshape(ny, nx)

    event_ids = np.sort(events_df["event_id"].unique())

    # Exact spatial order of the target array
    full_space = pd.DataFrame({
        "flat_index": np.arange(ny * nx),
        "grid": grid.reshape(-1),
        "lat": np.repeat(lat, nx),
        "lon": np.tile(lon, ny),
    })


    # Drop lat/lon from events_df for later duplication
    events_no_xy = events_df.drop(columns=["lat", "lon"], errors="ignore")

    # Create a full DataFrame of all events and all grid points
    full_events = (
        pd.DataFrame({"event_id": event_ids})
        .merge(full_space, how="cross")
    )
    events_df_full = full_events.merge(
        events_no_xy,
        on=["event_id", "grid"],
        how="left",
        validate="one_to_one",
        sort=False,
    )

    # Order the dataframe by event_id and flat_index for reshaping
    events_df_full = (
        events_df_full
        .sort_values(["event_id", "flat_index"])
        .reset_index(drop=True)
    )

    # Fill missing values with 0 for the FIELDS
    events_df_full[FIELDS] = events_df_full[FIELDS].fillna(0)

    # Validate the shape of the full DataFrame
    assert len(events_df_full) == len(event_ids) * ny * nx
    assert events_df_full.groupby("event_id")["grid"].nunique().eq(ny * nx).all()
    assert not events_df_full.duplicated(["event_id", "grid"]).any()

    if "day_of_event" in events_df.columns:
        events_df_full["day_of_event"] = events_df["day_of_event"].fillna(pd.Timedelta(days=9999))


    # Reshape the data into the desired format for training
    X = events_df_full[FIELDS].to_numpy().reshape([num_events, ny, nx, len(FIELDS)])

    D = events_df_full["day_of_event"].to_numpy(dtype="float32").reshape(num_events, ny, nx)

    U0 = events_df_full[[f"ecdf_{f}" for f in FIELDS]].to_numpy().reshape(num_events, ny, nx, len(FIELDS))

    U1 = events_df_full[[f"scdf_{f}" for f in FIELDS]].to_numpy().reshape(num_events, ny, nx, len(FIELDS))

    z = events_df_full[["event_id", "event_rp"]].groupby("event_id").mean().to_numpy().reshape(num_events)

    s = events_df[["event_id", "event_duration"]].groupby("event_id").mean().to_numpy().reshape(num_events)

    # Reshape the parameters for each field into a 4D array
    transform_params = (
        [f"thresh_{var}" for var in FIELDS]
        + [f"shape_{var}" for var in FIELDS]
        + [f"scale_{var}" for var in FIELDS]
    )

    gdf_params = (
        events_df_full[[*transform_params, "lon", "lat"]]
        .groupby(["lat", "lon"])
        .mean()
        .reset_index()
    )

    num_event_days_p0 = (
        events_df_full[["lat", "lon", "p_0_num_event_days"]]
        .groupby(["lat", "lon"])
        .mean()["p_0_num_event_days"]
        .to_numpy()
        .reshape(ny, nx)
    )

    param_grid = gdf_params.sort_values(["lat", "lon"])

    thresh = param_grid[[f"thresh_{f}" for f in FIELDS]].to_numpy().reshape(ny, nx, len(FIELDS))
    scale  = param_grid[[f"scale_{f}" for f in FIELDS]].to_numpy().reshape(ny, nx, len(FIELDS))
    shape  = param_grid[[f"shape_{f}" for f in FIELDS]].to_numpy().reshape(ny, nx, len(FIELDS))


    params = np.stack([thresh, scale, shape], axis=-2)

    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    # Combine all the data into an xarray Dataset and save to NetCDF
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------

    # Create an xarray Dataset to store the data
    coords = {
        "lat": lat,
        "lon": lon,
        "event_id": event_ids,
        "field": ["max_temp", "max_neg_spi", "num_event_days"],
        "param": ["loc", "scale", "shape"],
    }
    attrs = {
        "CRS": "EPSG:4326",
        "max_temp": "Maximum Temperature [°C]",
        "max_neg_spi": "Maximum Negative SPI [unitless]",
        "num_event_days": "Number of Event Days [days]",
        "project": "hazGAN",
        "note": "Fixed interpolation: [0, 1] --> (0, 1)."
    }

    ds = xr.Dataset({
        "uniform":      (["event_id", "lat", "lon", "field"], U1),
        "ecdf":         (["event_id", "lat", "lon", "field"], U0),
        "anomaly":      (["event_id", "lat", "lon", "field"], X),
        "day_of_event": (
                            ["event_id", "lat", "lon"],
                            D,
                            {
                                "units": "days",
                                "long_name": "days since event start",
                                "_FillValue": np.float32(9999),
                            }
                        ),
        "event_rp":     (["event_id"], z),
        "duration":     (["event_id"], s),
        "params":       (["lat", "lon", "param", "field"], params),
        "p_0_num_event_days": (["lat", "lon"], num_event_days_p0),
        "grid":         (["lat", "lon"], grid),
    }, coords=coords, attrs=attrs)

    ds.to_netcdf(f"{OUTPUT_DIR}/data.nc")


if __name__ == "__main__":
    main()