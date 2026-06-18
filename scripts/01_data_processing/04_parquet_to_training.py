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
from hazGAN.utils import res2str

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


def apply_colormap(grayscale_array, colormap_name="Spectral_r"):
    """Apply a colormap to a grayscale array and return an RGB uint8 array."""

    normalized = grayscale_array.astype(float) / 255
    colormap = plt.get_cmap(colormap_name)
    colored = colormap(normalized)
    rgb_uint8 = np.uint8(colored[..., :3] * 255)

    return rgb_uint8


def create_image_grid(image_paths, grid_size=(8, 8), output_path="grid.png"):
    """Create a grid of images from the provided image paths and save it to output_path.
    image_paths: list of paths to images
    grid_size: tuple (rows, cols) specifying the grid size
    output_path: path to save the resulting grid image
    """

    if len(image_paths) == 0:
        print(f"No images found for grid: {output_path}")
        return None

    with Image.open(image_paths[0]) as img:
        tile_width, tile_height = img.size

    total_width = tile_width * grid_size[1]
    total_height = tile_height * grid_size[0]

    output_img = Image.new("RGB", (total_width, total_height), "white")

    n_images = min(len(image_paths), grid_size[0] * grid_size[1])

    for idx in range(n_images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]

        x = col * tile_width
        y = row * tile_height

        with Image.open(image_paths[idx]) as img:
            output_img.paste(img.convert("RGB"), (x, y))

    output_img.save(output_path)
    print(f"Grid saved to: {output_path}")
    return output_img



# ----------------------------------------------
# ----------------------------------------------
# Main code
# ----------------------------------------------
# ----------------------------------------------


def main():

    # Load the parquet files
    events_df = pd.read_parquet(FOOTPRINTS_PARQUET_DIR / "events_jja.parquet")
    event_long = pd.read_parquet(FOOTPRINTS_PARQUET_DIR / "event_footprints_long_jja.parquet")


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

    plot_gpd_fits(raw_extra, var_list = ["max_temp", "max_neg_spi", "num_event_days"], save_dir=f"{FOOTPRINTS_PARQUET_DIR}/plots")


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
    events_df["day_of_event"] = events_df["time_max_temp"] - events_df["event_start"]
    events_df["day_of_event"] = pd.to_timedelta(events_df["day_of_event"])


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

    D = events_df_full["day_of_event"].values.reshape(num_events, ny, nx)

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
        "day_of_storm": (["event_id", "lat", "lon"], D),
        "storm_rp":     (["event_id"], z),
        "duration":     (["event_id"], s),
        "params":       (["lat", "lon", "param", "field"], params),
        "grid":         (["lat", "lon"], grid),
    }, coords=coords, attrs=attrs)

    ds.to_netcdf(f"{OUTPUT_DIR}/data.nc")

    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    # From training to images
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------


    DOMAIN = "gumbel"  # ["uniform", "gumbel", "gaussian"]
    EPS = 1e-6
    CMAP = "Spectral_r"

    # New dataset settings (image size)
    RES = (64, 64)
    EVENT_DIM = "event_id"

    # Choose filtering logic
    FILTER_MODE = "none"      # ["storm_rp", "field_max", "none"]
    FILTER_FIELD = "storm_rp"     # used only if FILTER_MODE == "field_max"
    FIELD_THRESHOLD = 300.0       # e.g. Kelvin if max_temp is absolute temp
    RP_THRESHOLD = 1           # keep events above this RP

    # Open dataset
    data_path = os.path.join(OUTPUT_DIR, "data.nc")
    try:
        ds = xr.open_dataset(data_path)
    except ValueError as e:
        if "Failed to decode variable 'day_of_storm'" in str(e):
            print("Detected day_of_storm decode conflict. Retrying with decode_timedelta=False...")
            ds = xr.open_dataset(data_path, decode_timedelta=False)
            if "day_of_storm" in ds:
                ds["day_of_storm"].attrs.pop("dtype", None)
            try:
                ds = xr.decode_cf(ds, decode_times=True, decode_timedelta=True)
            except Exception as decode_err:
                print(f"Warning: CF timedelta decode still failed: {decode_err}")
                print("Proceeding with raw day_of_storm values.")
        else:
            raise


    # Filter events based on the chosen mode
    if FILTER_MODE == "field_max":
        ds["event_filter_value"] = (
            ds["anomaly"]
            .sel(field=FILTER_FIELD)
            .max(dim=["lon", "lat"], skipna=True)
        )

        mask = ds["event_filter_value"] > FIELD_THRESHOLD

    elif FILTER_MODE == "storm_rp":
        mask = ds["storm_rp"] > RP_THRESHOLD
        pass

    elif FILTER_MODE == "none":
        mask = xr.ones_like(ds[EVENT_DIM], dtype=bool)

    else:
        raise ValueError(f"Unknown FILTER_MODE: {FILTER_MODE}")

    # Select only the events that pass the filter
    idx = np.where(mask.values)[0]
    ds = ds.isel({EVENT_DIM: idx})

    print(f"\nFound {ds[EVENT_DIM].size} events after filtering")

    # Define output directories for images
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_root = os.path.join(OUTPUT_DIR, "images", DOMAIN)
    winddir = os.path.join(img_root, "first_channel")
    stormdir = os.path.join(img_root, "rgb")

    os.makedirs(winddir, exist_ok=True)
    os.makedirs(stormdir, exist_ok=True)

    nimgs = ds[EVENT_DIM].size

    # Extract the uniform values and reshape for image processing
    array = ds["uniform"].transpose(EVENT_DIM, "lat", "lon", "field").values

    # Flip latitude so north appears at top
    array = np.flip(array, axis=1)

    # Handle NaNs safely
    array = np.where(np.isfinite(array), array, 0.0)

    # Assert that the uniform values are within the expected range [0, 1]
    if not ((np.nanmax(array) <= 1.0) and (np.nanmin(array) >= 0.0)):
        raise ValueError("Uniform values are not in [0, 1] range")

    # Validate the shape of the array
    ny = ds.sizes["lat"]
    nx = ds.sizes["lon"]
    nfields = ds.sizes["field"]
    assert array.shape[1:] == (ny, nx, nfields), f"Unexpected shape: {array.shape}"

    if nfields != 3:
        raise ValueError(f"Expected 3 fields for RGB image, found {nfields}")


    # Transform the uniform values to Gumbel space if required
    if DOMAIN == "gumbel":
        array = np.clip(array, EPS, 1 - EPS)
        array = -np.log(-np.log(array))

        array_min = np.nanmin(array, axis=(0, 1, 2), keepdims=True)
        array_max = np.nanmax(array, axis=(0, 1, 2), keepdims=True)

        denom = array_max - array_min
        denom = np.where(denom == 0, 1.0, denom)

        n = len(array)

        # Scale to approximately (0, 1), avoiding exact 0/1
        array = (array - array_min) / denom
        array = (array * (n - 1) + 1) / (n + 1)

        print("Gumbel image range:", np.nanmin(array), np.nanmax(array))
        print("Stats shape:", array_min.shape, array_max.shape)

        stats_path = os.path.join(img_root, "image_stats.npz")
        np.savez(stats_path, min=array_min, max=array_max, n=n)

    # Create and save images for each event
    for i in range(nimgs):
        arr = array[i]
        arr = np.where(np.isfinite(arr), arr, 0.0)
        arr = np.clip(arr, 0, 1)

        arr_uint8 = np.uint8(arr * 255)

        # Visualise first channel separately
        first_channel = arr_uint8[..., 0]
        colored_array = apply_colormap(first_channel, CMAP)
        colored_img = Image.fromarray(colored_array)
        colored_img = colored_img.resize((64, 64), Image.BILINEAR)
        output_path = os.path.join(winddir, f"event_{i}.png")
        colored_img.save(output_path)

        img = Image.fromarray(arr_uint8, "RGB")
        img = img.resize((64, 64), Image.BILINEAR)
        output_path = os.path.join(stormdir, f"event_{i}.png")
        img.save(output_path)

    storm_paths = sorted(glob.glob(os.path.join(stormdir, "event_*.png")))
    wind_paths = sorted(glob.glob(os.path.join(winddir, "event_*.png")))

    # Create image grids for visualization
    create_image_grid(
        storm_paths,
        grid_size=(8, 8),
        output_path=os.path.join(img_root, "percentiles_rgb.png"),
    )

    create_image_grid(
        wind_paths,
        grid_size=(8, 8),
        output_path=os.path.join(img_root, "percentiles_first_channel.png"),
    )
    
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    # Create anomaly images for training
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------

    # Define output directories for anomaly images
    anomaly_root = os.path.join(OUTPUT_DIR, "images", "anomaly")
    first_channel_dir = os.path.join(anomaly_root, "first_channel")
    rgb_dir = os.path.join(anomaly_root, "rgb")

    os.makedirs(first_channel_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    # Extract the anomaly values and reshape for image processing
    array = ds["anomaly"].transpose(EVENT_DIM, "lat", "lon", "field").values
    array = np.flip(array, axis=1)

    # Handle NaNs safely
    array = np.where(np.isfinite(array), array, np.nan)

    # Validate the shape of the array
    ny = ds.sizes["lat"]
    nx = ds.sizes["lon"]
    nfields = ds.sizes["field"]

    assert array.shape[1:] == (ny, nx, nfields), f"Unexpected shape: {array.shape}"

    if nfields != 3:
        raise ValueError(f"Expected 3 fields for RGB image, found {nfields}")

    # Global per-channel normalisation
    mins = np.nanmin(array, axis=(0, 1, 2), keepdims=True)
    maxs = np.nanmax(array, axis=(0, 1, 2), keepdims=True)

    denom = maxs - mins
    denom = np.where(denom == 0, 1.0, denom)

    array_norm = (array - mins) / denom
    array_norm = np.where(np.isfinite(array_norm), array_norm, 0.0)
    array_norm = np.clip(array_norm, 0, 1)

    # Save the min and max values for later use
    np.savez(
        os.path.join(anomaly_root, "anomaly_image_stats.npz"),
        min=mins,
        max=maxs,
    )

    # Create and save anomaly images for each event
    for i in range(ds[EVENT_DIM].size):
        arr = array_norm[i]
        arr_uint8 = np.uint8(arr * 255)

        first_channel = arr_uint8[..., 0]
        colored_array = apply_colormap(first_channel, CMAP)
        colored_img = Image.fromarray(colored_array)
        colored_img.save(os.path.join(first_channel_dir, f"event_{i}.png"))

        img = Image.fromarray(arr_uint8, "RGB")
        img.save(os.path.join(rgb_dir, f"event_{i}.png"))

    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "event_*.png")))
    first_channel_paths = sorted(glob.glob(os.path.join(first_channel_dir, "event_*.png")))

    print(f"{len(rgb_paths)} anomaly event images processed")

    # Create image grids for anomaly images
    create_image_grid(
        rgb_paths,
        grid_size=(8, 8),
        output_path=os.path.join(anomaly_root, "anomaly_rgb.png"),
    )

    # Create image grid for first channel anomaly images
    create_image_grid(
        first_channel_paths,
        grid_size=(8, 8),
        output_path=os.path.join(anomaly_root, "anomaly_first_channel.png"),
    )

if __name__ == "__main__":
    main()