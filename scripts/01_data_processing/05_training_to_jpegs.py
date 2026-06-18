import os
import warnings
from pathlib import Path

import numpy as np
import glob

warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
from dotenv import load_dotenv
from PIL import Image
import matplotlib.pyplot as plt


load_dotenv()

OUTPUT_DIR = Path(os.environ["TRAINING_DIR"])

# ----------------------------------------------
# Plot function
# ----------------------------------------------

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

def apply_colormap(grayscale_array, colormap_name="Spectral_r"):
    """Apply a colormap to a grayscale array and return an RGB uint8 array."""

    normalized = grayscale_array.astype(float) / 255
    colormap = plt.get_cmap(colormap_name)
    colored = colormap(normalized)
    rgb_uint8 = np.uint8(colored[..., :3] * 255)

    return rgb_uint8


# ----------------------------------------------
# ----------------------------------------------
# Main code
# ----------------------------------------------
# ----------------------------------------------


def main():

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