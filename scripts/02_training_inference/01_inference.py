"""
This script searches the samples directory for generated images and the training directory
for training data and parameters to convert images back to original scale.
"""
# %%
# quick defaults
TRAINRES = 64
STEP     = 300
MODEL    = "00019-images-low_shot-kimg300-color-translation-cutout"
WD       = "/data/ncas1/tb261/"


import os
import glob
import torch
import argparse
import numpy as np
import xarray as xr
from PIL import Image
from environs import Env
from numpy import asarray
import matplotlib.pyplot as plt
from hazGAN.statistics import invPIT, invPITDataset
from torchvision.transforms.functional import resize



def apply_colormap(grayscale_array, colormap_name='Spectral_r'):
    normalized = grayscale_array.astype(float) / 255
    colormap = plt.get_cmap(colormap_name)
    colored = colormap(normalized)
    rgb_uint8 = np.uint8(colored[..., :3] * 255)
    return rgb_uint8


def create_image_grid(image_paths, grid_size=(32, 32), output_path="grid.png"):
    with Image.open(image_paths[0]) as img:
        tile_width, tile_height = img.size
    total_width = tile_width * grid_size[1]
    total_height = tile_height * grid_size[0]
    output_img = Image.new('RGB', (total_width, total_height), 'white')
    n_images = min(len(image_paths), grid_size[0] * grid_size[1])
    for idx in range(n_images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        x = col * tile_width
        y = row * tile_height
        with Image.open(image_paths[idx]) as img:
            output_img.paste(img, (x, y))
    output_img.save(output_path)
    print(f"Grid saved to: {output_path}")
    return output_img


# script begins here
if __name__ == "__main__":
    print("\nReading environment variables...")
    # process args
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', dest="WD", type=str, default=WD, help='Training runs directory.')
    parser.add_argument('--model', '-m', dest="MODEL", type=str, default=MODEL)
    parser.add_argument('--step', '-s', dest='STEP', type=int, default=STEP)
    parser.add_argument('--train-res', '-t', dest='TRAINRES', type=int, default=TRAINRES, help="Training data's original resolution.")
    parser.add_argument("--res", '-r', type=int, default=64, choices=[16, 32, 64, 128, 256, 512], help="Sample resolution")

    args, unknown = parser.parse_known_args()

    WD       = args.WD
    MODEL    = args.MODEL
    TRAINRES = args.TRAINRES
    STEP     = str(args.STEP).zfill(6)
    RES      = args.res

    print(f"Making results images for resolution: {RES} pixels.")

    # source training data
    datadir = os.path.join(WD, "training")
    datadir = os.path.join(datadir, f"{TRAINRES}x{TRAINRES}_jja_0_1_2")

    # model run directories
    resultsdir = os.path.join(WD, "stylegan_events/training-runs/", MODEL, "results")
    indir      = os.path.join(resultsdir, "trunc-1_0")
    winddir    = os.path.join(resultsdir, "samples_single_field")
    griddir    = os.path.join(resultsdir, "grids")
    percdir    = os.path.join(winddir, "percentiles")
    quantdir   = os.path.join(winddir, "quantiles")

    os.makedirs(winddir, exist_ok=True)
    os.makedirs(percdir, exist_ok=True)
    os.makedirs(quantdir, exist_ok=True)
    os.makedirs(griddir, exist_ok=True)

    imlist = glob.glob(os.path.join(indir, "seed*.png"))
    print(f"Found {len(imlist)} images in {indir}.")

    # load generated images and convert to numpy
    # load generated images and convert to numpy
print("Loading generated images...")

samples = []

for i, impath in enumerate(sorted(imlist)):
    image = Image.open(impath)
    array = asarray(image).astype(np.float32) / 255.0

    assert array.shape == (RES, RES, 3), (
        f"Unexpected shape: {array.shape} != {(RES, RES, 3)}"
    )

    if array.min() < 0 or array.max() > 1:
        raise ValueError("Image values outside [0, 1]", array.min(), array.max())

    samples.append(array)

    # Save raw generated first-channel image
    first_channel = np.uint8(array[..., 0] * 255)
    colored_array = apply_colormap(first_channel)
    Image.fromarray(colored_array).save(os.path.join(percdir, f"seed{i}.png"))

fake_img = np.stack(samples, axis=0)

print(f"Loaded generated images: {fake_img.shape}")


# load training samples
print("\nLoading training data...")

data = xr.open_dataset(os.path.join(datadir, "data.nc"))

params = data["params"].values
p_0_gridcell = data["p_0_num_event_days"].values
train_x = data["anomaly"].values

data_lat = data["lat"].values if "lat" in data.coords else np.arange(train_x.shape[1])
data_lon = data["lon"].values if "lon" in data.coords else np.arange(train_x.shape[2])

# Match historical orientation used by plotting / inverse transform
params = np.flip(params, axis=0)
p_0_gridcell = np.flip(p_0_gridcell, axis=0)
train_x = np.flip(train_x, axis=1)
data_lat = np.flip(data_lat)


# resize generated data to match training resolution
if fake_img.shape[1:3] != train_x.shape[1:3]:
    print("Generated images shape:", fake_img.shape)
    print("Training data shape:", train_x.shape)
    print("Resizing generated images...")

    fake_img = torch.tensor(fake_img).permute(0, 3, 1, 2)
    fake_img = resize(fake_img, (TRAINRES, TRAINRES))
    fake_img = fake_img.permute(0, 2, 3, 1).numpy()

    print("Resized to:", fake_img.shape)
else:
    print("Not resizing images.")

# build xarray objects for inverse PIT
field_names = ["max_temp", "max_neg_spi", "num_event_days"]

fake_gumbel_img = xr.Dataset(
    data_vars={
        "gumbel": (["sample", "lat", "lon", "field"], fake_img)
    },
    coords={
        "sample": np.arange(fake_img.shape[0]),
        "lat": data_lat,
        "lon": data_lon,
        "field": field_names,
    },
)


# undo training image scaling: PNG [0, 1] -> Gumbel -> uniform
print("Undoing Gumbel image scaling...")

stats_file = os.path.join(datadir, "images", "gumbel", "image_stats.npz")
stats = np.load(stats_file)

image_minima = stats["min"]
image_maxima = stats["max"]
n = stats["n"]
image_range = image_maxima - image_minima

fake_gumbel = (
    (fake_img * (n + 1) - 1) / (n - 1)
    * image_range
    + image_minima
)

fake_u = np.exp(-np.exp(-fake_gumbel))
fake_u = np.clip(fake_u, 1e-6, 1 - 1e-6)


fake_ds = xr.Dataset(
    data_vars={
        "uniform": (["sample", "lat", "lon", "field"], fake_u)
    },
    coords={
        "sample": np.arange(fake_u.shape[0]),
        "lat": data_lat,
        "lon": data_lon,
        "field": field_names,
    },
)

train_x_ds = xr.Dataset(
    data_vars={
        "anomaly": (["time", "lat", "lon", "field"], train_x)
    },
    coords={
        "time": np.arange(train_x.shape[0]),
        "lat": data_lat,
        "lon": data_lon,
        "field": field_names,
    },
)

params_ds = xr.Dataset(
    data_vars={
        "params": (["lat", "lon", "param", "field"], params)
    },
    coords={
        "lat": data_lat,
        "lon": data_lon,
        "param": ["loc", "scale", "shape"],
        "field": field_names,
    },
)

ds_inv = xr.Dataset(
    data_vars={
        "uniform": fake_ds["uniform"],
        "anomaly": train_x_ds["anomaly"],
        "gumbel": fake_gumbel_img["gumbel"],
        "params": params_ds["params"],
        "p_0_num_event_days": (["lat", "lon"], p_0_gridcell),
    }
)


# inverse PIT to original scale
print("Applying inverse PIT...")

fake = invPITDataset(
    ds_inv,
    theta_var="params",
    u_var="uniform",
    x_var="anomaly",
    p0_var="p_0_num_event_days",
    gumbel_margins=False,
)


# save back-transformed samples
print("Saving back-transformed samples to NetCDF...")

generated_ds = fake.astype(np.float64).to_dataset(name="generated")

generated_ds = generated_ds.assign_attrs({
    "model": MODEL,
    "step": STEP,
    "train_resolution": TRAINRES,
    "sample_resolution": RES,
    "description": "Back-transformed GAN samples on original variable scale",
})

generated_ds = xr.Dataset(
    data_vars={
        "generated_png_scale": (
            ["sample", "lat", "lon", "field"],
            fake_img.astype(np.float32),
        ),
        "generated_gumbel": (
            ["sample", "lat", "lon", "field"],
            fake_gumbel.astype(np.float32),
        ),
        "generated_uniform": (
            ["sample", "lat", "lon", "field"],
            fake_u.astype(np.float32),
        ),
        "generated_backtransformed": (
            ["sample", "lat", "lon", "field"],
            fake.values.astype(np.float64),
        ),
    },
    coords={
        "sample": np.arange(fake_img.shape[0]),
        "lat": data_lat,
        "lon": data_lon,
        "field": field_names,
    },
    attrs={
        "model": MODEL,
        "step": STEP,
        "train_resolution": TRAINRES,
        "sample_resolution": RES,
    },
)

out_nc = os.path.join(
    resultsdir,
    f"generated_all_scales_{RES}x{RES}.nc",
)
generated_ds.to_netcdf(out_nc)

print(f"Saved numeric back-transformed samples to: {out_nc}")


# save single-channel visualisations of backtransformed field 0
print("Saving single-channel backtransformed images...")

fake_scaled = (
    (fake - fake.min(dim=("lat", "lon"))) /
    (fake.max(dim=("lat", "lon")) - fake.min(dim=("lat", "lon")))
)

fake_scaled = fake_scaled.fillna(0).clip(0, 1).values

for i in range(fake_scaled.shape[0]):
    array = np.uint8(fake_scaled[i] * 255)
    first_channel = array[..., 0]

    colored_array = apply_colormap(first_channel)
    Image.fromarray(colored_array).save(os.path.join(quantdir, f"seed{i}.png"))


# make grids
print("Creating grids...")

perc_paths = sorted(glob.glob(os.path.join(percdir, "seed*.png")))
quant_paths = sorted(glob.glob(os.path.join(quantdir, "seed*.png")))

perc_paths = [path for path in perc_paths if "grid" not in path]
quant_paths = [path for path in quant_paths if "grid" not in path]

if len(perc_paths) > 0:
    create_image_grid(
        perc_paths,
        (8, 8),
        os.path.join(griddir, f"p{RES}x{RES}_jja.png"),
    )

if len(quant_paths) > 0:
    create_image_grid(
        quant_paths,
        (8, 8),
        os.path.join(griddir, f"q{RES}x{RES}_jja.png"),
    )

print("Done!")