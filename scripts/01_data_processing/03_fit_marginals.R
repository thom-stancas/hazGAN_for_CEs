rm(list = ls())
.libPaths(c("~/R/libs/hazGAN", .libPaths()))
library(arrow)
library(dotenv)

# Load environment variables from .env file
load_dot_env(".env")

ROOT_DIR <- Sys.getenv("ROOT_DIR")
setwd(ROOT_DIR)
PARQUETDIR <- Sys.getenv("PARQUETDIR")
VERSION <- Sys.getenv("VERSION")

source("scripts/01_data_processing/utils/utils.R")
source("scripts/01_data_processing/utils/settings.R")

WD      <- Sys.getenv("ERA5DIR")
DRYRUN  <- FALSE
NDRYRUN <- 1000

#%%######## LOAD FOOTPRINT DATA ################################################

# Read NetCDF files matching the pattern
data <- read_parquet(file.path(PARQUETDIR, paste0("event_footprints_long_jja_", VERSION, ".parquet")))

# Optional dry run: keep only a the first NDRYRUN grids (for testing) so that we fit the dist across all events but only on a subset of the data
if (DRYRUN) {
  print(paste0("Dry run: keeping only the first ", NDRYRUN, " grids for testing..."))
  data <- data %>%
    filter(grid %in% unique(grid)[1:NDRYRUN])
}

#%%######## TRANSFORM EVENTS ###################################################
print("Tranforming fields...")

hot_dry_temp <- gpd_transformer(data, "max_temp", Q); warnings()
hot_dry_spi  <- gpd_transformer(data, "max_neg_spi", Q); warnings()

# empirical-only for event days
hot_dry_event_days <- empirical_transformer(data, "num_event_days"); warnings()

#%%######## PUT TOGETHER #######################################################
print("Done. Putting it all together...")

renamer <- function(df, var) {
  df <- df %>%
    rename_with(~ paste0(., ".", var),
                -c("grid", "event_id", "event.rp", "lat", "lon", "variable"))
  df <- df %>%
    rename_with(~ var, "variable")
  return(df)
}

hot_dry_temp       <- renamer(hot_dry_temp, "max_temp")
hot_dry_spi        <- renamer(hot_dry_spi, "max_neg_spi")
hot_dry_event_days <- renamer(hot_dry_event_days, "num_event_days")

events <- hot_dry_temp %>%
  inner_join(hot_dry_spi, by = c("grid", "event_id", "event.rp", "lat", "lon")) %>%
  inner_join(hot_dry_event_days, by = c("grid", "event_id", "event.rp", "lat", "lon"))

events$thresh.q <- Q  # keep track of threshold used

#%%######## SAVE RESULTS #######################################################
if (!DRYRUN) {
  print("Saving...")
  out_file <- file.path(PARQUETDIR, paste0("events_jja_", VERSION, ".parquet"))
  write_parquet(events, out_file)
  cat("\nSaved as:", out_file)
  print(paste0("Finished! ", length(unique(events$event_id)), " events processed."))
} else {
  print("Saving dry run...")
  out_file <- file.path(PARQUETDIR, "events_dryrun.parquet")
  write_parquet(events, out_file)
  cat("\nSaved as:", out_file)
  print(paste0("Finished! ", length(unique(events$event_id)), " events processed."))
}

#%%######## END ################################################################