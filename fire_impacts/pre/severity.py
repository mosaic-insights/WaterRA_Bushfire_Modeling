"""
Calculate fire severity (NBR and dNBR) from DEA satellite imagery.

Queries the Digital Earth Australia STAC catalogue for pre- and
post-fire imagery, computes the Normalised Burn Ratio (NBR) and the
delta-NBR (dNBR), and saves rasters and metadata to the project's
FireSeverity folder.
"""

import os
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import pystac_client
import odc.stac
from dea_tools.bandindices import calculate_indices
import logging
from .project import FireImpactsProject
from .util import metres_to_approx_degrees, clip_raster
from fire_impacts import util as toputil
from . import mask_dnbr as mdnbr
from fire_impacts.util import date_rel
from .data_sources import (
    DEA_STAC,
    DEA_STATUS_URL,
    SENTINEL_2_COLLECTIONS,
    LANDSAT_COLLECTIONS,
)

logger = logging.getLogger(__name__)

CATALOG = None

# Imagery before this date uses Landsat; on/after it uses Sentinel-2.
SPLIT_DATE = pd.Timestamp("2016-07-01")


# ---------------------------------------------------------------------------
# STAC catalogue initialisation
# ---------------------------------------------------------------------------

def init_catalog(url: str = DEA_STAC):
    """
    Connect to the DEA STAC API and initialise the module-level catalog.

    Parameters:
    - url: STAC API endpoint URL (defaults to DEA_STAC).

    Returns:
    - None.  Sets the module-level CATALOG variable.
    """
    global CATALOG
    if CATALOG is None:
        try:
            CATALOG = pystac_client.Client.open(url)
        except Exception as e:
            logger.error(
                "Failed to connect to STAC API at %s: %s", url, e
            )
            if url == DEA_STAC:
                logger.error(
                    "This is the default Digital Earth Australia (DEA) "
                    "STAC URL. DEA may be experiencing issues. Check "
                    "%s and try again later.",
                    DEA_STATUS_URL,
                )
            raise RuntimeError(
                f"Could not initialize STAC catalog from {url}"
            ) from e
        odc.stac.configure_rio(
            cloud_defaults=True,
            aws={"aws_unsigned": True},
        )


# ---------------------------------------------------------------------------
# NBR computation
# ---------------------------------------------------------------------------

def calc_nbr(
    datetime,
    label,
    bbox,
    filter_query,
    desired_crs,
    desired_resolution,
    use_mask=True,
):
    """
    Compute median NBR for an area over a date range from DEA imagery.

    Queries the DEA STAC catalogue, loads Sentinel-2 or Landsat ARD
    data (automatically split at the sensor changeover date), normalises
    band names, and returns the time-median NBR as an xarray DataArray.
    Requires the module-level CATALOG to be initialised via
    init_catalog() before calling.

    Parameters:
    - datetime: date range string 'YYYY-MM-DD/YYYY-MM-DD'.
    - label: human-readable label used in log messages.
    - bbox: bounding box (minx, miny, maxx, maxy) in the target CRS.
    - filter_query: STAC filter expression string (e.g. cloud cover).
    - desired_crs: output CRS for the loaded imagery.
    - desired_resolution: output pixel resolution.
    - use_mask: if True, load the s2cloudless cloud mask for Sentinel-2.

    Returns:
    - image_metadata: list of dicts describing each image used.
    - nbr_ds: xarray DataArray of median NBR values, or None if no
      imagery was found.
    """
    if CATALOG is None:
        raise RuntimeError(
            "CATALOG is not initialised. "
            "Call init_catalog() before calc_nbr()."
        )

    # Parse the incoming datetime string: 'YYYY-MM-DD/YYYY-MM-DD'
    dt0_str, dt1_str = datetime.split("/")
    dt_start = pd.Timestamp(dt0_str)
    dt_end = pd.Timestamp(dt1_str)

    # Safety check in case of inverted ranges
    if dt_end < dt_start:
        raise ValueError(f"datetime range is invalid: {datetime}")

    # Build subranges: Landsat before SPLIT_DATE, Sentinel-2 after
    subranges = []

    # Part strictly before SPLIT_DATE >> Landsat
    if dt_start < SPLIT_DATE:
        lt_start = dt_start
        lt_end = min(dt_end, SPLIT_DATE - pd.Timedelta(seconds=1))
        if lt_start <= lt_end:
            subranges.append(("landsat", lt_start, lt_end))

    # Part on/after SPLIT_DATE >> Sentinel-2
    if dt_end >= SPLIT_DATE:
        s2_start = max(dt_start, SPLIT_DATE)
        s2_end = dt_end
        if s2_start <= s2_end:
            subranges.append(("sentinel", s2_start, s2_end))

    # For each subrange, query STAC, load, and normalise band names
    all_items = []
    ds_parts = []

    for sensor, d0, d1 in subranges:
        # Format a STAC datetime range for this subrange
        dt_str = f"{d0:%Y-%m-%d}/{d1:%Y-%m-%d}"

        if sensor == "sentinel":
            # Use the package-level SENTINEL_2_COLLECTIONS for Sentinel-2
            collections = SENTINEL_2_COLLECTIONS
            bands = ["nbart_nir_1", "nbart_swir_3"]
            # Optionally add the cloud mask for pre-fire runs
            if use_mask:
                bands.append("oa_s2cloudless_mask")
        else:
            # Landsat: use LANDSAT_COLLECTIONS defined above
            collections = LANDSAT_COLLECTIONS
            bands = ["nbart_nir", "nbart_swir_2"]
            # Note: no s2cloudless mask for Landsat

        logger.info(
            "calc_nbr(): loading %s data for %s (%s to %s)",
            sensor, label, d0.date(), d1.date(),
        )

        # Search for datasets for the requested area and date
        query = CATALOG.search(
            bbox=bbox,
            collections=collections,
            datetime=dt_str,
            filter=filter_query,
            sortby="eo:cloud_cover",
        )
        items = list(query.items())
        logger.info(
            "Found %d %s datasets for %s", len(items), sensor, label
        )

        if not items:
            continue  # no data for this sensor / subrange

        # STAC load for this subrange and sensor
        ds = odc.stac.load(
            items,
            bands=bands,
            crs=desired_crs,
            resolution=desired_resolution,
            groupby="solar_day",
            bbox=bbox,
        )

        # Normalise band names so that downstream NBR formula is generic.
        # We map everything onto 'nir' and 'swir'.
        rename_map = {}
        if sensor == "sentinel":
            if "nbart_nir_1" in ds.data_vars:
                rename_map["nbart_nir_1"] = "nir"
            if "nbart_swir_3" in ds.data_vars:
                rename_map["nbart_swir_3"] = "swir"
        else:
            if "nbart_nir" in ds.data_vars:
                rename_map["nbart_nir"] = "nir"
            if "nbart_swir_2" in ds.data_vars:
                rename_map["nbart_swir_2"] = "swir"

        if rename_map:
            ds = ds.rename(rename_map)

        # In rare cases filtering can result in no time steps left
        if ds.sizes.get("time", 0) == 0:
            logger.warning(
                "calc_nbr(): %s dataset for %s has 0 timesteps "
                "after load; skipping.",
                sensor, label,
            )
            continue

        ds_parts.append(ds)
        all_items.extend(items)

    # Combine all sensor parts; if nothing loaded, bail out
    if not ds_parts:
        logger.warning(
            "calc_nbr(): no imagery found for %s over the interval %s.",
            label, datetime,
        )
        # Keep return types consistent: empty metadata, None raster
        return [], None

    # Concatenate along time dimension and sort by time
    ds_all = xr.concat(ds_parts, dim="time").sortby("time")

    # Extract metadata for all images actually used
    image_metadata = extract_image_metadata(
        all_items,
        ds_all["time"].values,
        desired_resolution,
        label,
    )

    # Compute NBR using normalised 'nir' and 'swir' bands
    # NBR = (NIR - SWIR) / (NIR + SWIR)
    if "nir" not in ds_all.data_vars or "swir" not in ds_all.data_vars:
        raise KeyError(
            "calc_nbr(): expected 'nir' and 'swir' bands after "
            "normalisation."
        )

    nir = ds_all["nir"]
    swir = ds_all["swir"]

    # Avoid division-by-zero issues; skipna=True ignores bad pixels.
    nbr = (nir - swir) / (nir + swir)

    # Take the median along time
    image = nbr.median(dim="time", skipna=True)

    return image_metadata, image


# ---------------------------------------------------------------------------
# Raster output helper
# ---------------------------------------------------------------------------

def write_raster_xarray(
    data,
    crs,
    path,
    name,
    extent_shape=None,
):
    """
    Write an xarray DataArray to a GeoTIFF, optionally clipping to a
    shapefile extent.

    Parameters:
    - data: xarray DataArray with rioxarray accessor.
    - crs: CRS to write to the output raster.
    - path: directory to write the output file.
    - name: base filename (without extension) for the output GeoTIFF.
    - extent_shape: optional path to a shapefile; if supplied, the
      raster is clipped to its extent before saving.

    Returns:
    - None.  Writes <name>.tif to path.
    """
    # Write a temporary raster with the CRS set
    data_with_crs = data.rio.write_crs(crs)
    tmp_path = os.path.join(path, "tmp.tif")
    data_with_crs.rio.to_raster(tmp_path)

    # Construct the desired output file name
    out_path = os.path.join(path, f"{name}.tif")

    # If a file with the same name already exists, remove it
    if os.path.exists(out_path):
        logger.info(
            "severity.write_raster_array() is overwriting a file at %s.",
            out_path,
        )
        os.remove(out_path)

    # Clip the output to the extent shape if desired
    if extent_shape is not None:
        final_path, _ = clip_raster(tmp_path, extent_shape)
        os.remove(tmp_path)
    else:
        final_path = tmp_path

    # Rename the raster to the final output name
    os.rename(final_path, out_path)


# ---------------------------------------------------------------------------
# Fire severity pipeline
# ---------------------------------------------------------------------------

def calculate_fire_severity(
    project: FireImpactsProject,
    fire_start_date,
    fire_end_date,
    start_date_pre=None,
    end_date_post=None,
    max_cloud_cover=20,
    resolution_input=20,
    bbox=None,
    catchment=None,
):
    """
    Calculate fire severity (NBR, dNBR) before and after the fire.

    Loads pre- and post-fire imagery from the DEA STAC catalogue,
    computes NBR for each period, saves both NBR rasters, computes
    dNBR, and writes a masked dNBR using the DEA Land Cover layer.

    Parameters:
    - project: FireImpactsProject with catchments already loaded.
    - fire_start_date: date string for the start of the fire.
    - fire_end_date: date string for the end of the fire.
    - start_date_pre: start date for pre-fire imagery; defaults to
      90 days before fire_start_date.
    - end_date_post: end date for post-fire imagery; defaults to
      90 days after fire_end_date.
    - max_cloud_cover: maximum cloud cover percentage filter.
    - resolution_input: pixel resolution in metres.
    - bbox: bounding box for the catchment area; calculated from the
      catchment boundary if not supplied.
    - catchment: name of the catchment to process; if None, processes
      all catchments in the project.

    Returns:
    - None.  Saves NBR, dNBR, and metadata files to the catchment's
      FireSeverity folder.
    """
    # If no catchment is specified, run for each catchment in the project
    if catchment is None:
        return project.for_each_catchment(
            lambda c: calculate_fire_severity(
                project,
                fire_start_date,
                fire_end_date,
                start_date_pre,
                end_date_post,
                max_cloud_cover,
                resolution_input,
                bbox,
                catchment=c,
            )
        )

    # Initialise the STAC catalog if it hasn't been done already
    if CATALOG is None:
        init_catalog()

    # Get the last day before the fire, and the first day after it
    end_date_pre = date_rel(fire_start_date, -1)
    start_date_post = date_rel(fire_end_date, 1)

    # If the user hasn't specified a pre-fire period, default to 90 days
    if start_date_pre is None:
        logger.info(
            "No start date for pre-fire data provided. "
            "Defaulting to 90 days before fire start date."
        )
        start_date_pre = date_rel(fire_start_date, -90)

    # If the user hasn't specified a post-fire period, default to 90 days
    if end_date_post is None:
        logger.info(
            "No end date for post-fire data provided. "
            "Defaulting to 90 days after fire end date."
        )
        end_date_post = date_rel(fire_end_date, 90)

    if bbox is None:
        logger.info(
            "Bounding box not provided. Calculating from shapefile "
            "with 10 km buffer."
        )
        bbox = project.catchment_bounds(catchment, 10)

    # Define the filter for cloud cover
    filter_query = f"eo:cloud_cover < {max_cloud_cover}"

    # Load the catchment shapefile and prepare the output folder
    catchment_folder = project.catchment_path(catchment, "FireSeverity")
    os.makedirs(catchment_folder, exist_ok=True)
    shapefile_path = project.boundary_files[catchment]
    gdf = gpd.read_file(shapefile_path)

    # If the catchment is in a geographic CRS, convert the resolution
    # (metres) to an approximate equivalent in degrees.
    if gdf.crs.axis_info[0].unit_name == "degree":
        alt_resolution_input = metres_to_approx_degrees(resolution_input)
        logger.info(
            "Alternative resolution (degrees): %f (was %fm)",
            alt_resolution_input, resolution_input,
        )
        resolution_input = alt_resolution_input

    # Arguments common to both pre- and post-fire NBR calls
    common_args = {
        "bbox": bbox,
        "filter_query": filter_query,
        "desired_crs": gdf.crs,
        "desired_resolution": resolution_input,
    }
    nbr_label = "NBR"
    dnbr_label = "d" + "NBR"

    pre_fire_label = "Prefire"
    pre_fire_name = f"{pre_fire_label}_{nbr_label}"

    # Calculate Pre-fire NBR
    logger.info("Calculating pre-fire NBR for %s", catchment)
    pre_image_metadata, prefire_NBR = calc_nbr(
        datetime=f"{start_date_pre}/{end_date_pre}",
        label=pre_fire_label,
        **common_args,
        use_mask=True,
    )

    # Write the pre-fire NBR to disk
    logger.info("Writing pre-fire NBR raster...")
    write_raster_xarray(
        prefire_NBR, gdf.crs, catchment_folder, pre_fire_name, shapefile_path
    )

    post_fire_label = "Postfire"
    post_fire_name = f"{post_fire_label}_{nbr_label}"

    # Calculate Post-fire NBR
    logger.info("Calculating post-fire NBR for %s", catchment)
    post_image_metadata, postfire_NBR = calc_nbr(
        datetime=f"{start_date_post}/{end_date_post}",
        label=post_fire_label,
        **common_args,
        use_mask=False,
    )

    # Write the post-fire NBR to disk
    write_raster_xarray(
        postfire_NBR,
        gdf.crs,
        catchment_folder,
        post_fire_name,
        shapefile_path,
    )

    # Save combined pre- and post-fire image metadata to CSV
    combined_metadata = pre_image_metadata + post_image_metadata
    save_metadata_to_csv(
        combined_metadata,
        catchment_folder,
        "Satellite_image_information_combined.csv",
    )

    # Calculate dNBR = pre-fire NBR minus post-fire NBR
    logger.info("Calculating dNBR for %s", catchment)
    delta_NBR = prefire_NBR - postfire_NBR
    # Set negative dNBR values to 0
    delta_NBR = delta_NBR.where(delta_NBR >= 0, 0)
    # Save fire date metadata
    fire_meta = pd.DataFrame(
        data={
            "Value": [
                pd.to_datetime(fire_start_date),
                pd.to_datetime(fire_end_date),
            ]
        },
        index=["start_date", "end_date"],
    )
    fire_meta.index.name = "Key"
    fire_path = os.path.join(catchment_folder, "FireMeta.csv")
    fire_meta.to_csv(fire_path, date_format="%Y-%m-%d")
    logger.info("Saved fire metadata to %s", fire_path)

    # Write the dNBR raster to the catchment folder
    delta_fire_label = "dNBR"
    write_raster_xarray(
        delta_NBR,
        gdf.crs,
        catchment_folder,
        delta_fire_label,
        shapefile_path,
    )

    # Mask dNBR to retain only naturally-vegetated pixels
    mdnbr.mask_dnbr(project=project, catchment=catchment)

    logger.info("Processes are completed")


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def extract_image_metadata(items, valid_times, resolution_input, fire_status):
    """
    Build a list of metadata dicts for a set of STAC items.

    Parameters:
    - items: list of pystac Item objects from a STAC search.
    - valid_times: array of numpy datetime64 values representing the
      times loaded into the xarray dataset.
    - resolution_input: pixel resolution used for the load call.
    - fire_status: label string (e.g. 'Prefire', 'Postfire').

    Returns:
    - List of dicts with satellite name, product, time, cloud cover,
      and resolution; only items whose datetime is in valid_times are
      included.
    """
    valid_times_str = [
        pd.to_datetime(time).isoformat() + "Z" for time in valid_times
    ]
    metadata = [
        {
            "Pre/Post-fire": fire_status,
            "Satellite Name": item.properties.get("platform"),
            "Product ID": item.collection_id,
            "Time of Image": item.properties.get("datetime"),
            "Cloud Cover (%)": round(
                item.properties.get("eo:cloud_cover"), 1
            ),
            "Resolution": resolution_input,
        }
        for item in items
        if item.properties.get("datetime") in valid_times_str
    ]
    return metadata


def save_metadata_to_csv(metadata, folder, filename):
    """
    Save a list of metadata dicts to a CSV file in a given folder.

    Parameters:
    - metadata: list of dicts to convert to a DataFrame and save.
    - folder: directory path for the output CSV.
    - filename: filename (with extension) for the output CSV.

    Returns:
    - None.  Writes the CSV to os.path.join(folder, filename).
    """
    pd.DataFrame(metadata).to_csv(os.path.join(folder, filename))