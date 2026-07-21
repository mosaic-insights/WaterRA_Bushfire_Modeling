"""
Classes and functions for managing fire impacts project folder
structure and data.
"""

import os
from glob import glob
from pathlib import Path
import shutil
import rasterio as rio
import numpy as np
import geopandas as gpd
import pandas as pd
import json
import logging
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from .. import util as toputil
from .. import const
from ..run_context import EventRunContext

logger = logging.getLogger(__name__)

# Sentinel used to distinguish "kwarg omitted" from an explicit None
# in APIs where None has its own meaning (e.g. "clear the registered
# value" vs. "keep what's there").
_UNSET = object()

# Default directories required inside every catchment directory.
PER_CATCHMENT_FOLDERS = const.PER_CATCHMENT_FOLDERS

STATS = const.STATS
APPROX_KM_PER_DEGREE = const.APPROX_KM_PER_DEGREE

# State exactly what dtypes we're happy to save rasters in.
default_dtypes_raster = {
    'int': rio.int32,
    'float': rio.float32,
    }

# Convert numpy one-character dtype.kind attributes into more
# general descriptors that map into default_dtypes_raster.
numpy_kind_to_desc = const.numpy_kind_to_desc


###############################################################################
####### FireImpactsProject ####################################################
###############################################################################
class FireImpactsProject(object):
    """
    Represents the project folder structure for a fire impacts study.
    --------------------------------------------------------------------
    Notes:
    - Keeps track of data related to one or more catchments.
    - Register catchments using add_catchment() or
      add_all_catchments().
    --------------------------------------------------------------------
    """

    # --- Persistence --------------------------------------------------------

    ###########################################################################
    def __init__(self, project_path, exist_ok=False, clear=False):
        """
        Initialise a project from a project path.

        Parameters:
        - project_path: Path to the project folder.
        - exist_ok: If True, do not raise an error if the project
          folder already exists.
        - clear: If True, clear the project folder if it already
          exists.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        norm_path = os.path.normpath(project_path)
        self.project_path = norm_path
        self.catchments = []
        self.boundary_files = {}
        self.source_data = {}
        # Per-catchment name of the string field in the subcatchment
        # layer used as the label in downstream outputs
        # (e.g. 'SiteID' for Avon). Populated by add_subcatchments()
        # or set_subcatchment_label_field() and persisted in settings.
        self.subcatchment_label_fields: dict = {}

        # If the user has said to clear the existing folder OR they
        # have said to proceed even if there is already a folder:
        if clear or not exist_ok:
            self.initialise_project(
                norm_path, exist_ok=exist_ok, clear=clear
                )
        else:
            try:
                self.load_project()
            except:
                self.initialise_project(
                    norm_path, exist_ok=exist_ok, clear=clear
                    )

        self.load_vis_defaults()
        self.load_name_defaults()

    ###########################################################################
    def _settings_fn(self):
        """
        Return the path to the project's settings.json file.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        return os.path.join(self.project_path, 'settings.json')

    ###########################################################################
    def _settings(self):
        """
        Return the settings dict to be written to settings.json.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        settings_dict = dict(
            catchments=self.catchments,
            source_data=self.source_data,
            boundary_files=self.boundary_files,
            subcatchment_label_fields=self.subcatchment_label_fields,
            )
        return settings_dict

    ###########################################################################
    def _write(self):
        """
        Write current project settings to settings.json.
        ----------------------------------------------------------------
        Notes:
        - Saves paths for boundary files in a JSON-friendly format.
        ----------------------------------------------------------------
        """
        with open(self._settings_fn(), 'w') as f:
            json.dump(self._settings(), f, indent=2)

    # --- Path helpers -------------------------------------------------------

    ###########################################################################
    def catchment_path(self, catchment_name=None, *args):
        """
        Build a path relative to a particular catchment's folder.

        Parameters:
        - catchment_name: Name of the catchment. If not provided,
          returns the top-level Catchments folder path.
        - args: Additional sub-path components below the catchment
          folder (e.g. 'Erodibility', 'KLSCP.tif').

        Returns:
        - Full path to the catchment folder or sub-path.
        ----------------------------------------------------------------
        Notes:
        - Args should correspond to subfolder names; for example
          'Erodibility', 'KLSCP.tif' gives the full path to that
          file.
        ----------------------------------------------------------------
        """
        # Every project will have a Catchments folder:
        base = os.path.join(self.project_path, 'Catchments')
        if catchment_name is None:
            assert len(args) == 0, (
                'Cannot specify additional arguments without a '
                'catchment name.'
                )
            return base
        return os.path.join(base, catchment_name, *args)

    ###########################################################################
    def ensemble_path(
        self,
        catchment_name: str,
        *args,
        event: str = 'default',
        ensemble: str = 'default',
        ):
        """
        Resolve a path under a catchment's event + ensemble folder.

        Parameters:
        - catchment_name: Name of the catchment.
        - args: Path components appended below the ensemble folder.
        - event: Event name. Defaults to 'default' so single-event
          projects can ignore this parameter entirely.
        - ensemble: Ensemble name within the event. Defaults to
          'default'.

        Returns:
        - Full path under Catchments/<catchment>/Events/<event>/
          Ensemble/<ensemble>/<args>.
        ----------------------------------------------------------------
        Notes:
        - Multiple ensembles per event support comparing the same
          fire under current vs. future climate.
        - The Events/ layer is the forward-compatible seam for
          planned multi-event support.
        ----------------------------------------------------------------
        """
        return os.path.join(
            self.catchment_path(catchment_name),
            'Events', event, 'Ensemble', ensemble,
            *args,
            )

    # --- Project initialisation and loading ---------------------------------

    ###########################################################################
    def load_project(self):
        """
        (Re)load project settings from settings.json.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        with open(self._settings_fn(), 'r') as f:
            settings = json.load(f)
        self.catchments = settings.get('catchments', [])
        self.source_data = settings.get('source_data', {})
        self.boundary_files = settings.get('boundary_files', {})
        self.subcatchment_label_fields = settings.get(
            'subcatchment_label_fields', {}
            )
        self.ensure_catchment_folders()
        self.load_vis_defaults()
        self.load_name_defaults()

    ###########################################################################
    def initialise_project(
        self, project_path, exist_ok=False, clear=False
        ):
        """
        Create a new project at the given path.

        Parameters:
        - project_path: Desired location, with the project name as
          the final folder component.
        - exist_ok: If True, allow creation inside an existing folder.
        - clear: If True, remove project-managed entries
          (settings.json, Catchments/) before re-initialising.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        # If there is already a folder and the user has said NOT to
        # clear it:
        if not clear and os.path.exists(project_path):
            raise FileExistsError(
                f'Project folder already exists: {project_path}'
                )
        # If there is already a folder and the user has said it is ok
        # to clear its contents. Only remove project-managed entries
        # (settings.json and Catchments/) rather than the entire
        # folder so that initialising into an existing directory
        # (e.g. '.') is safe and doesn't blow away unrelated files.
        if clear and os.path.exists(project_path) and not exist_ok:
            logger.info(
                f'Clearing project entries (settings.json, '
                f'Catchments/) in: {project_path}'
                )
            settings_path = self._settings_fn()
            if os.path.isfile(settings_path):
                os.remove(settings_path)
            catchments_dir = self.catchment_path()
            if os.path.isdir(catchments_dir):
                shutil.rmtree(catchments_dir)
        # Create a new Catchments folder and write empty settings:
        os.makedirs(self.catchment_path(), exist_ok=exist_ok)
        self._write()

    # --- Catchment registration ---------------------------------------------

    ###########################################################################
    def add_catchment(
        self,
        catchment_shapefile: str | Path,
        name=None,
        replace_existing=False,
        subcatchment_id_cols: list = None,
        subcatchment_label_field=_UNSET,
        ):
        """
        Register a new catchment in the project.

        Parameters:
        - catchment_shapefile: Shapefile of the catchment boundary.
          When this file contains multiple polygons it is treated as
          a subcatchment coverage: geometries are dissolved to a
          single boundary and the original file is registered via
          add_subcatchments().
        - name: Catchment name. Defaults to the shapefile basename.
        - replace_existing: If True, replace an existing catchment
          of the same name. Otherwise raise.
        - subcatchment_id_cols: Forwarded to add_subcatchments()
          when the source shapefile is dissolved into a boundary and
          a subcatchment coverage. Ignored for single-polygon inputs.
        - subcatchment_label_field: Forwarded to add_subcatchments()
          in the same circumstances as subcatchment_id_cols.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        # If a name hasn't been specified, derive one from the
        # shapefile name:
        if name is None:
            name = os.path.splitext(
                os.path.basename(catchment_shapefile)
                )[0]

        # Check if the catchment is already registered:
        have_already = name in self.catchments
        if have_already and not replace_existing:
            raise ValueError(
                f'Catchment {name} already exists in project.'
                )
        if not have_already:
            self.catchments.append(name)

        # Inspect the input: a multi-feature shapefile is interpreted
        # as a subcatchment coverage. Dissolve to a single boundary
        # for the catchment itself; the original file is then
        # registered as the subcatchment layer below.
        src_gdf = gpd.read_file(catchment_shapefile)
        is_coverage = len(src_gdf) > 1

        # Make sure the catchment folder structure exists before we
        # write any derived files into it.
        self.ensure_catchment_folders(name)

        if is_coverage:
            dissolved = src_gdf.dissolve()
            boundary_path = os.path.join(
                self.catchment_path(name), f'{name}_boundary.shp'
                )
            dissolved.to_file(boundary_path)
            self.boundary_files[name] = boundary_path
            logger.info(
                f'Source shapefile {catchment_shapefile} contains '
                f'{len(src_gdf)} features; dissolved to a single '
                f'boundary at {boundary_path}. The original coverage '
                f'will be registered as subcatchments.'
                )
        else:
            self.boundary_files[name] = str(catchment_shapefile)

        # Update settings.json with the new catchment:
        self._write()

        # When the source was a coverage, register it as the
        # subcatchment layer. This must run after the boundary is in
        # place - add_subcatchments() clips against it.
        if is_coverage:
            self.add_subcatchments(
                name,
                str(catchment_shapefile),
                id_cols=subcatchment_id_cols or [],
                label_field=subcatchment_label_field,
                )

    ###########################################################################
    def add_subcatchments(
        self,
        catchment_name: str,
        subcatch_shapefile_path: str,
        id_cols: list = [],
        label_field=_UNSET,
        ):
        """
        Load subcatchments for a catchment from a shapefile.

        Parameters:
        - catchment_name: Name of the catchment to attach
          subcatchments to.
        - subcatch_shapefile_path: Path to the subcatchment shapefile.
        - id_cols: Attribute columns from the source shapefile to
          retain alongside the internal sc_ID index.
        - label_field: Which of the retained columns to treat as the
          preferred string label for downstream outputs (e.g.
          'SiteID').
        ----------------------------------------------------------------
        Notes:
        - label_field has three-way semantics:
          - Omitted: keeps any existing registration if the field
            is still present, else defaults to the first id_col
            that is a string column.
          - Explicit None: clears any registered label field.
          - String: sets the field explicitly; warning logged if
            it replaces a prior registration.
        - The resolved label is persisted in settings.json so that
          helpers like combine_rusle_and_debris_subcatchment() can
          pick it up automatically.
        - Reprojects to the catchment CRS if needed.
        - Clips to the catchment boundary.
        - Retains identifying attributes from id_cols.
        - Saves the processed shapefile in the Subcatchments folder
          and updates settings.json.
        ----------------------------------------------------------------
        """
        in_gdf = gpd.read_file(subcatch_shapefile_path)

        # Check and compare CRS of subcatchment and existing
        # catchment; reproject if needed:
        subcatch_crs = in_gdf.crs
        catch_crs = self.catchment_crs(catchment_name)
        if subcatch_crs != catch_crs:
            catch_epsg = catch_crs.to_epsg()
            subcatch_epsg = subcatch_crs.to_epsg()
            logger.info(
                f'Subcatchment shapefile CRS is EPSG: {subcatch_epsg}.'
                f' Reprojecting to catchment CRS (EPSG: {catch_epsg}).'
                )
            int_gdf = in_gdf.to_crs(catch_crs)
        else:
            int_gdf = in_gdf

        # Clip the subcatchments to the catchment boundary:
        catch_gdf = self.catchment_boundary(catchment_name)
        logger.info(
            'Clipping subcatchments to the catchment polygon...'
            )
        subcatch_clipped = int_gdf.clip(catch_gdf)

        # Raise an error if there is no shared area:
        if subcatch_clipped.empty:
            raise ValueError(
                'Only subcatchment areas within the catchment '
                'boundary can be processed, but there were none '
                'left after clipping.'
                )

        # Register the source shapefile path in boundary_files:
        key_name = catchment_name + '_' + 'subcatchments'
        previous_source = self.boundary_files.get(key_name)
        previous_label = self.subcatchment_label_fields.get(
            catchment_name
            )
        if (previous_source is not None
                and previous_source != subcatch_shapefile_path):
            logger.warning(
                f"Replacing registered subcatchments for catchment "
                f"'{catchment_name}': source was {previous_source}, "
                f"now {subcatch_shapefile_path}. The saved clipped "
                f"shapefile will be overwritten."
                )
        self.boundary_files[key_name] = subcatch_shapefile_path

        # Resolve the label field with three-way semantics:
        # omitted (sentinel) -> preserve existing;
        # explicit None -> clear registration; string -> use as-is.
        if label_field is _UNSET:
            if previous_label is not None:
                if previous_label in in_gdf.columns:
                    logger.warning(
                        f"add_subcatchments() called for catchment "
                        f"'{catchment_name}' without label_field=, "
                        f"but '{previous_label}' is already "
                        f"registered and present in "
                        f"{subcatch_shapefile_path} - keeping it. "
                        f"Pass label_field=None to clear."
                        )
                    resolved_label = previous_label
                else:
                    logger.warning(
                        f"add_subcatchments() called for catchment "
                        f"'{catchment_name}' without label_field=, "
                        f"and the previously registered field "
                        f"'{previous_label}' is not present in "
                        f"{subcatch_shapefile_path}. Subcatchment "
                        f"outputs will fall back to integer indices."
                        )
                    resolved_label = None
            elif id_cols:
                first = id_cols[0]
                if (first in in_gdf.columns
                        and in_gdf[first].dtype == object):
                    resolved_label = first
                else:
                    resolved_label = None
            else:
                resolved_label = None
        else:
            # Caller specified explicitly (either a name or None):
            resolved_label = label_field
            if (resolved_label is not None
                    and previous_label is not None
                    and resolved_label != previous_label):
                logger.warning(
                    f"Changing subcatchment label field for "
                    f"catchment '{catchment_name}' from "
                    f"'{previous_label}' to '{resolved_label}'."
                    )

        if resolved_label is not None:
            self.subcatchment_label_fields[catchment_name] = (
                resolved_label
                )
        else:
            self.subcatchment_label_fields.pop(catchment_name, None)

        self._write()

        # Get only the useful columns plus geometry. Always retain
        # the resolved label column so downstream code can label
        # outputs by subcatchment name rather than integer index.
        good_cols = list(id_cols)
        if (resolved_label is not None
                and resolved_label not in good_cols):
            if resolved_label in subcatch_clipped.columns:
                good_cols.append(resolved_label)
        good_cols.append(subcatch_clipped.geometry.name)
        int_gdf = subcatch_clipped[good_cols]

        # Use the index as the internal integer subcatchment id
        # (sc_ID):
        out_gdf = int_gdf.reset_index(
            drop=False, names=self.subcatchment_id
            )

        # Save the clipped subcatchments to the Subcatchments folder:
        save_path = self.catchment_path(
            catchment_name, 'Subcatchments'
            )
        key_file_name = key_name + '.shp'
        key_file_path = os.path.join(save_path, key_file_name)
        out_gdf.to_file(key_file_path)
        logger.info(
            f'Saved clipped subcatchments shapefile to {key_file_path}'
            )

    ###########################################################################
    def add_all_catchments(self, catchment_shapefiles):
        """
        Register all catchments in the project from a list of
        shapefiles.

        Parameters:
        - catchment_shapefiles: List of paths to the shapefiles
          defining the catchment boundaries.
        ----------------------------------------------------------------
        Notes:
        - Replaces any existing catchments with the same names.
        ----------------------------------------------------------------
        """
        for shapefile in catchment_shapefiles:
            logger.info(f'Adding catchment from: {shapefile}')
            self.add_catchment(shapefile, replace_existing=True)

    # --- Subcatchment label field -------------------------------------------

    ###########################################################################
    def subcatchment_label_field(self, catchment_name: str):
        """
        Return the preferred subcatchment label field, or None.

        Parameters:
        - catchment_name: Name of the catchment to look up.

        Returns:
        - The registered label field name, or None if not set.
        ----------------------------------------------------------------
        Notes:
        - Set via add_subcatchments() (label_field= argument) or
          set_subcatchment_label_field().
        - Consumed by helpers like
          combine_rusle_and_debris_subcatchment() so that output
          columns carry meaningful names without per-call config.
        ----------------------------------------------------------------
        """
        return self.subcatchment_label_fields.get(catchment_name)

    ###########################################################################
    def set_subcatchment_label_field(
        self, catchment_name: str, field: str | None,
        ):
        """
        Set or clear the preferred subcatchment label field.

        Parameters:
        - catchment_name: Name of the catchment to update.
        - field: Field name to register, or None to clear.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        # None clears the registration; a string is validated against
        # the subcatchment columns before being set:
        if field is None:
            self.subcatchment_label_fields.pop(catchment_name, None)
        else:
            subs = self.get_subcatchments(catchment_name)
            if field not in subs.columns:
                raise ValueError(
                    f"Field '{field}' not found on "
                    f"{catchment_name} subcatchments. Available: "
                    f"{list(subs.columns)}"
                    )
            self.subcatchment_label_fields[catchment_name] = field
        self._write()

    # --- Directory management -----------------------------------------------

    ###########################################################################
    def ensure_catchment_folders(self, catchment_name: str = None):
        """
        Create required catchment sub-folders if they don't exist.

        Parameters:
        - catchment_name: Name of the catchment to check. If not
          provided, runs for all registered catchments.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        # If no name given, recurse for every registered catchment:
        if catchment_name is None:
            for catchment in self.catchments:
                self.ensure_catchment_folders(catchment)
            return
        catchment_path = self.catchment_path(catchment_name)
        # Create each standard subfolder if it doesn't already exist:
        for folder in PER_CATCHMENT_FOLDERS:
            os.makedirs(
                os.path.join(catchment_path, folder), exist_ok=True
                )

    # --- Geometry access ----------------------------------------------------

    ###########################################################################
    def catchment_boundary(self, catchment: str) -> gpd.GeoDataFrame:
        """
        Get the catchment boundary as a GeoDataFrame.

        Parameters:
        - catchment: Name of the catchment.

        Returns:
        - GeoDataFrame of the catchment boundary file.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        shapefile_path = self.boundary_files[catchment]
        gdf = gpd.read_file(shapefile_path)
        return gdf

    ###########################################################################
    def get_subcatchments(self, catchment: str) -> gpd.GeoDataFrame:
        """
        Get the subcatchment boundaries as a GeoDataFrame.

        Parameters:
        - catchment: Name of the catchment.

        Returns:
        - GeoDataFrame of subcatchments if available, otherwise the
          catchment boundary itself.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        shape_name = catchment + '_subcatchments.shp'
        project_folder = 'Subcatchments'
        new_id_col_name = self.subcatchment_id
        gdf = self.get_catchment_polygons(
            catchment,
            project_folder,
            shape_name,
            new_id_col_name,
            )
        return gdf

    ###########################################################################
    def get_headwaters(self, catchment: str) -> gpd.GeoDataFrame:
        """
        Get the headwater boundaries as a GeoDataFrame.

        Parameters:
        - catchment: Name of the catchment.

        Returns:
        - GeoDataFrame of headwaters.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        shape_name = 'Headwaters.shp'
        project_folder = 'Topography'
        new_id_col_name = self.headwater_id
        gdf = self.get_catchment_polygons(
            catchment,
            project_folder,
            shape_name,
            new_id_col_name,
            )
        return gdf

    ###########################################################################
    def get_catchment_polygons(
        self,
        catchment: str,
        folder: str,
        poly_file_name: str,
        auto_id_col_name: str,
        ) -> gpd.GeoDataFrame:
        """
        Read a polygon shapefile from a catchment sub-folder.

        Parameters:
        - catchment: Name of the catchment.
        - folder: Sub-folder name within the catchment directory.
        - poly_file_name: Shapefile filename.
        - auto_id_col_name: Column name to assign from the row index
          if not already present in the shapefile.

        Returns:
        - GeoDataFrame with all shapefile columns, plus the auto-id
          column if it was absent.
        ----------------------------------------------------------------
        Notes:
        - The auto-id column is only added when not already present.
          extract_headwaters() writes hw_ID as 1-based integers;
          blindly overwriting it with gdf.index (0-based) would cause
          a mismatch when merging against any CSV built from the
          shapefile's own IDs.
        ----------------------------------------------------------------
        """
        shapefile_path = self.catchment_path(
            catchment, folder, poly_file_name
            )
        if os.path.exists(shapefile_path):
            gdf = gpd.read_file(shapefile_path)
            if auto_id_col_name not in gdf.columns:
                gdf[auto_id_col_name] = gdf.index
            return gdf

        raise FileNotFoundError(
            f'Catchment polygons ({poly_file_name}) were requested '
            f'from project.get_catchment_polygons() for {catchment}, '
            'but they appear not to be loaded yet. Use '
            'project.add_subcatchments() or '
            'topography.extract_headwaters() first.'
            )

    # --- Catchment properties -----------------------------------------------

    ###########################################################################
    def catchment_bounds(
        self, catchment: str, buffer_distance_km: float = 10
        ):
        """
        Get the WGS84 bounding box for a catchment with an optional
        buffer.

        Parameters:
        - catchment: Name of the catchment.
        - buffer_distance_km: Kilometres beyond the catchment
          boundary to buffer before computing the bounding box.

        Returns:
        - List of [min_lon, min_lat, max_lon, max_lat] for the
          buffered boundary.
        ----------------------------------------------------------------
        Notes:
        - Buffer accommodates projection differences and is primarily
          used when requesting satellite data through dea-tools.
        ----------------------------------------------------------------
        """
        gdf = self.catchment_boundary(catchment)
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        bbox = gdf_wgs84.total_bounds

        # Convert km to approximate degrees (1 degree ≈ 111 km):
        buffer_degrees = buffer_distance_km / APPROX_KM_PER_DEGREE

        bbox_with_buffer = [
            bbox[0] - buffer_degrees,  # minx with buffer
            bbox[1] - buffer_degrees,  # miny with buffer
            bbox[2] + buffer_degrees,  # maxx with buffer
            bbox[3] + buffer_degrees,  # maxy with buffer
            ]
        return bbox_with_buffer

    ###########################################################################
    def catchment_crs(self, catchment: str):
        """
        Get the CRS for a catchment from its boundary shapefile.

        Parameters:
        - catchment: Name of the catchment.

        Returns:
        - GeoPandas CRS object.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        gdf = self.catchment_boundary(catchment)
        return gdf.crs

    ###########################################################################
    def cell_area(self, catchment: str = None):
        """
        Get the cell area of the DEM for a catchment.

        Parameters:
        - catchment: Name of the catchment. If not provided,
          returns results for all catchments.

        Returns:
        - Planar area of one DEM cell in the DEM's native units, or
          a dict of values keyed by catchment name.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        if catchment is None:
            return self.for_each_catchment(
                lambda c: self.cell_area(catchment=c)
                )
        fn = self.catchment_path(catchment, 'Topography', 'DEM.tif')
        with rio.open(fn) as src:
            transform = src.transform
            # transform.a is x pixel size; transform.e is y pixel
            # size (negative for north-up rasters):
            return abs(transform.a * transform.e)

    # --- Iteration ----------------------------------------------------------

    ###########################################################################
    def for_each_catchment(self, fn: callable):
        """
        Run a function for each catchment in the project.

        Parameters:
        - fn: Function that takes a catchment name and optionally
          returns a value.

        Returns:
        - Dict mapping catchment name to the function's return value.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        logger.info(f'Processing {len(self.catchments)} catchments')
        return {
            catchment: fn(catchment)
            for catchment in self.catchments
            }

    # --- Visualisation configuration ----------------------------------------

    ###########################################################################
    def load_vis_defaults(self):
        """
        Load default visualisation parameter dicts.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        from matplotlib.colors import LogNorm

        # Topographic layers:
        self.vis_DEM = {
            'cmap': 'viridis',
            'measure': 'Elevation',
            'units': 'm',
            'norm': None,
            'cbar_extend': 'neither',
            'title_varname': 'DEM',
            }

        self.vis_slope = {
            'cmap': 'viridis',
            'measure': 'Slope',
            'units': '°',
            'norm': None,
            'cbar_extend': 'neither',
            'title_varname': 'Slope',
            }

        self.vis_flow_accum = {
            'cmap': 'viridis',
            'measure': 'Contributing areas',
            'units': 'count',
            'norm': 'log',
            'vmin': 10,
            'cbar_extend': 'min',
            'title_varname': 'Flow Accumulation',
            }

        # Fire severity (dNBR) - two scales for raw vs. standardised:
        # Raw raster files use a 0-1 scale; extend min to show
        # negatives without clipping the upper end.
        self.vis_dNBR_raw = {
            'cmap': 'inferno',
            'measure': 'ΔNBR',
            'units': 'raw',
            'title_varname': 'ΔNBR',
            'norm': 'linear',
            'vmin': 0.0,
            'vmax': 1.0,
            'cbar_extend': 'min',
            }

        # Standardised dNBR (zonal stats columns): 0-1000 scale,
        # fixed range so the colour is consistent across catchments.
        self.vis_dNBR_std = {
            'cmap': 'inferno',
            'measure': 'ΔNBR',
            'units': 'standardised',
            'title_varname': 'ΔNBR',
            'norm': 'linear',
            'vmin': 0.0,
            'vmax': 1000.0,
            'cbar_extend': 'neither',
            }

        # Debris flow inputs and outputs:
        self.vis_i12_crit = {
            'cmap': 'plasma_r',
            'measure': '12-minute intensity threshold',
            'units': 'mm/hr',
            'title_varname': 'Rain Intensity I12 Crit',
            'norm': 'linear',
            'vmin': 0.0,
            'vmax': 800.0,
            'cbar_extend': 'max',
            }

        self.vis_num_debris_flow_events = {
            'cmap': 'Reds',
            'measure': 'Debris Flow Events',
            'units': 'count',
            'title_varname': 'Debris Flow Events',
            'norm': 'boundary',
            'cbar_extend': 'neither',
            }

        # Soil and climate factors:
        self.vis_aridity = {
            'cmap': 'cividis',
            'measure': 'Aridity Factor',
            'units': 'wet → dry',
            'title_varname': 'Aridity',
            'norm': 'linear',
            'cbar_extend': 'neither',
            }

        # Erosion and sediment delivery rasters (stored in t/cell;
        # converted to t/ha at plot time via scale_to_per_ha):
        self.vis_erosion = {
            'cmap': 'cividis',
            'measure': 'Erosion',
            'units': 't/ha',
            'title_varname': '',
            'norm': 'log',
            'cbar_extend': 'neither',
            # Rasters are stored in t/cell; convert to t/ha at
            # plot time using the actual raster cell size.
            'scale_to_per_ha': True,
            }

        self.vis_delivered = {
            'cmap': 'cividis',
            'measure': 'Sediment Delivery',
            'units': 't/ha',
            'title_varname': '',
            'norm': 'log',
            'cbar_extend': 'neither',
            # Rasters are stored in t/cell; convert to t/ha at
            # plot time using the actual raster cell size.
            'scale_to_per_ha': True,
            }

        # Debris mass (log scale; units are kg per cell):
        self.vis_debris_mass = {
            'cmap': 'cividis',
            'measure': 'Available Debris Mass',
            'units': 'Kg',
            'title_varname': 'Debris Flow Mass',
            'norm': 'log',
            'vmin': 0,
            'cbar_extend': 'neither',
            }

    ###########################################################################
    def get_vis_params(self, file_or_col_name: str):
        """
        Get appropriate visualisation parameters for a raster or
        column name.

        Parameters:
        - file_or_col_name: Name of the raster file or data column
          to look up.

        Returns:
        - Dict of visualisation parameters, falling back to defaults
          if no match is found.
        ----------------------------------------------------------------
        Notes:
        - dNBR is routed specially: raw raster files (dNBR.tif,
          masked_dNBR.tif) use the 0-1 raw scale; zonal-stat columns
          (dNBR_mean, dNBR_max, ...) use the 0-1000 standardised
          scale.
        ----------------------------------------------------------------
        """
        # Normalise the input string for case-insensitive lookup:
        input_string = (
            file_or_col_name.lower().strip().replace(' ', '_')
            )
        clay_mass_fmt = (
            const.DEBRIS_MASS_FIELD.lower().strip().replace(' ', '_')
            )

        default_params = {
            'cmap': 'viridis',
            'measure': 'Undefined',
            'units': 'n/a',
            'norm': None,
            'cbar_extend': 'neither',
            'title_varname': '',
            }

        # Special case: dNBR routes to raw or standardised vis params
        # depending on whether the input looks like a stats column:
        _stat_suffixes = ('_mean', '_max', '_min', '_median', '_std')
        if 'dnbr' in input_string:
            is_stats_col = any(
                input_string.endswith(s) for s in _stat_suffixes
                )
            return (
                self.vis_dNBR_std if is_stats_col else self.vis_dNBR_raw
                )

        # Map keyword substrings to vis_params dicts:
        param_dict = {
            'slope': self.vis_slope,
            'flow_acc': self.vis_flow_accum,
            'i12_crit': self.vis_i12_crit,
            'num_events': self.vis_num_debris_flow_events,
            'aridity': self.vis_aridity,
            'erosion': self.vis_erosion,
            'delivered': self.vis_delivered,
            'dem': self.vis_DEM,
            clay_mass_fmt: self.vis_debris_mass,
            'plain': default_params,
            }

        for key, value in param_dict.items():
            if key in input_string:
                return value

        logger.info(
            f'Visualisation parameters not found for '
            f'{file_or_col_name}. Falling back to defaults.'
            )
        return default_params

    ###########################################################################
    def load_name_defaults(self):
        """
        Load default field names from const.
        ----------------------------------------------------------------
        Notes:
        - This may no longer be needed with the const.py module.
        ----------------------------------------------------------------
        """
        self.headwater_id = const.HW_ID
        self.subcatchment_id = const.SC_ID

    # --- Plotting -----------------------------------------------------------

    ###########################################################################
    def plot_catchment_raster(
        self,
        *args,
        catchment=None,
        existing_figure=None,
        axes_index=None,
        new_subplot: bool = True,
        ):
        """
        Plot the requested raster for one or all catchments.

        Parameters:
        - args: Path components identifying the raster within the
          catchment folder.
        - catchment: Name of the catchment to plot. If None, one
          subplot per catchment is created.
        - existing_figure: matplotlib figure to plot onto. A new one
          is created if not provided.
        - axes_index: Index of the axes within the figure to draw on.
        - new_subplot: Whether to add a new subplot as part of this
          call.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        # Set up the figure, creating one if not provided. Track which
        # axes index to draw onto:
        if existing_figure is None:
            from matplotlib import pyplot as plt
            figure = plt.figure()
            if axes_index is None:
                axes_index = 0
        else:
            figure = existing_figure
            if axes_index is None:
                axes_index = len(figure.axes) - 1
            else:
                if new_subplot:
                    logger.warning(
                        f'project.plot_catchment_raster() received '
                        f'axes {axes_index} but a new subplot was '
                        f'also requested via new_subplot=True. This '
                        f'is contradictory and will most likely '
                        f'produce an undesired result, like plots '
                        f'partially overlapping.'
                        )

        # If no catchment is specified, create one subplot per
        # catchment in the project:
        if catchment is None:
            figure.subplots(
                nrows=len(self.catchments),
                ncols=1,
                )
            self.for_each_catchment(
                lambda c: self.plot_catchment_raster(
                    *args,
                    catchment=c,
                    existing_figure=figure,
                    axes_index=self.catchments.index(c),
                    new_subplot=False,
                    )
                )
            return
        else:
            if new_subplot:
                num_subs_already = len(figure.axes)
                figure.add_subplot(
                    num_subs_already + 1,
                    1,
                    num_subs_already + 1,
                    )

        import rasterio as rio
        import os
        import numpy as np

        # Resolve the raster path and look up visualisation params:
        raster_path = self.catchment_path(catchment, *args)
        if not raster_path.endswith('.tif'):
            raster_path += '.tif'

        gdf = self.catchment_boundary(catchment)
        file_name = args[-1]
        vis_params = self.get_vis_params(file_name)

        # Determine the chart title from the filename:
        useful_filename_part = file_name.split('.')[0].lower()
        if 'erosion' in file_name:
            title = toputil.get_erosion_title(
                useful_filename_part, 'erosion'
                )
            vis_params['title_varname'] = title
        elif 'delivered' in file_name:
            title = toputil.get_erosion_title(
                useful_filename_part, 'delivered'
                )
            vis_params['title_varname'] = title

        # Fix the colour scale for erosion/delivery rasters so that
        # year 1 and year 2 are always comparable. Peak (30-min) and
        # total use different bounds; delivered tends to be lower than
        # erosion so has its own set of limits:
        if 'erosion' in file_name:
            is_peak = 'peak' in file_name
            vis_params['vmin'] = 0.01 if is_peak else 10
            vis_params['vmax'] = 50 if is_peak else 1000
            vis_params['cbar_extend'] = 'both'
        elif 'delivered' in file_name:
            is_peak = 'peak' in file_name
            vis_params['vmin'] = 0.001 if is_peak else 0.1
            vis_params['vmax'] = 50 if is_peak else 500
            vis_params['cbar_extend'] = 'both'

        # Render the raster onto the axes and add boundary overlay:
        catch_name = toputil.clean_chart_title(catchment)
        chart_title = catch_name + ': ' + vis_params['title_varname']

        ax = figure.axes[axes_index]
        img, this_crs, cbar = toputil.plot_spatial_raster(
            ax,
            raster_path,
            vis_params,
            title=chart_title,
            colourbar=True,
            clip_geometry=gdf,
            )

        # Get the coordinate reference of the raster to extract
        # unit info for axis labels:
        if this_crs.is_projected:
            these_units = this_crs.linear_units + 's'
        elif this_crs.is_geographic:
            # Assumes degrees are the only relevant angular unit:
            these_units = 'degrees'

        toputil.mapify_axes(ax, this_crs, these_units)
        plot_catchment_boundary(self, catchment, ax)

    ###########################################################################
    def plot_catchment_polygons(
        self,
        catchment: str,
        polygons: gpd.GeoDataFrame,
        colour_col: str,
        vis_params: dict,
        title: str,
        non_geo_data: pd.DataFrame | None = None,
        id_col: str | None = None,
        existing_figure=None,
        existing_axes=None,
        ):
        """
        Plot catchment polygons optionally coloured by a data column.

        Parameters:
        - catchment: Name of the catchment (used to overlay the
          boundary line).
        - polygons: GeoDataFrame of polygons to plot.
        - colour_col: Column name to use for polygon colouring.
        - vis_params: Dict of visualisation parameters.
        - title: Axes title.
        - non_geo_data: Optional DataFrame with non-spatial data to
          join to the polygons for colouring.
        - id_col: Column name linking non_geo_data to polygons.
        - existing_figure: matplotlib figure to plot onto.
        - existing_axes: matplotlib axes to plot onto.
        ----------------------------------------------------------------
        Notes:
        - Specific callers (plot_headwaters, plot_subcatchments)
          should get the relevant GeoDataFrame and non-spatial data,
          then call this method to do the actual plotting.
        ----------------------------------------------------------------
        """
        fig, ax = toputil.fig_ax_admin(existing_figure, existing_axes)

        this_crs, cbar, ax = toputil.plot_spatial_vector(
            ax,
            polygons,
            vis_params,
            title,
            symbol_data=non_geo_data,
            id_col_name=id_col,
            data_col_name=colour_col,
            )

        # Set a grey background to aid readability:
        ax.set_facecolor('#D3D3D3')

        these_units = this_crs.axis_info[0].unit_name
        toputil.mapify_axes(ax, this_crs, these_units)
        plot_catchment_boundary(self, catchment, ax)

    ###########################################################################
    def plot_headwaters(
        self,
        catchment: str,
        colour_col: str | None = None,
        table: pd.DataFrame | None = None,
        data_type: str = '',
        existing_figure=None,
        existing_axes=None,
        ):
        """
        Plot headwaters coloured by a specified data value.

        Parameters:
        - catchment: Name of the catchment.
        - colour_col: Column name to use for colouring. If None
          (with no table provided), plots plain shapes.
        - table: Optional pre-loaded DataFrame. Skips file loading
          if provided.
        - data_type: Output type subfolder name, typically
          'DebrisFlow' (or '' for soil/slope summary).
        - existing_figure: matplotlib figure to plot onto.
        - existing_axes: matplotlib axes to plot onto.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        headwaters_gdf = self.get_headwaters(catchment)

        if data_type == 'DebrisFlow':
            data_folder = data_type
            data_file_name = 'DebrisFlowData'
        else:
            data_folder = None
            data_file_name = 'Soil_Slope_Aridity_dNBR_headwaters'

        # If no colour column or data table is provided, skip data
        # loading entirely and render plain shapes:
        if colour_col is None and table is None:
            non_geo_data = None
        else:
            non_geo_data = self.get_table_safely(
                colour_col=colour_col,
                data_type=data_folder,
                data_file=data_file_name,
                catchment=catchment,
                allow_basic=False,
                table=table,
                )

        if (non_geo_data is not None
                and colour_col in non_geo_data.columns):
            id_col = self.headwater_id
            ng_for_join = non_geo_data[[id_col, colour_col]]
            actual_colour_col = colour_col
            column_for_title = colour_col
        else:
            id_col = None
            ng_for_join = None
            actual_colour_col = None
            column_for_title = '(plain)'

        if colour_col is not None:
            vis_params = self.get_vis_params(colour_col)
        else:
            vis_params = self.get_vis_params('Plain')

        ax_title = toputil.make_axes_title(
            catchment,
            'Headwaters',
            vis_params['title_varname'],
            column_for_title,
            )

        self.plot_catchment_polygons(
            catchment=catchment,
            polygons=headwaters_gdf,
            colour_col=actual_colour_col,
            vis_params=vis_params,
            title=ax_title,
            non_geo_data=ng_for_join,
            id_col=id_col,
            existing_figure=existing_figure,
            existing_axes=existing_axes,
            )

    ###########################################################################
    def plot_subcatchments(
        self,
        catchment: str,
        colour_col: str,
        data_type: str | None = None,
        data_file: str | None = None,
        table: pd.DataFrame | None = None,
        existing_figure=None,
        existing_axes=None,
        ):
        """
        Plot subcatchment polygons coloured by a specified data
        column.

        Parameters:
        - catchment: Name of the catchment.
        - colour_col: Column name to use for polygon colouring.
        - data_type: Optional subfolder name under the catchment
          directory where the data CSV lives (e.g. 'Results',
          'DebrisFlow'). Auto-detected from colour_col if not given.
        - data_file: Optional CSV file name (without extension).
          Auto-detected from colour_col if not given.
        - table: Optional pre-loaded data table. Skips file loading
          if provided.
        - existing_figure: matplotlib figure to plot onto.
        - existing_axes: matplotlib axes to plot onto.
        ----------------------------------------------------------------
        Notes:
        - Auto-detection rules (when data_file is not supplied):
            - colour_col contains 'erosion' or 'delivered':
              reads rusle_subcatchment_summary.csv from Results/
            - colour_col contains 'events', 'debris', 'mass', or
              'i12': reads DebrisFlowData_subcatchments.csv from
              DebrisFlow/
            - otherwise: reads Soil_Slope_Aridity_dNBR_subcatchments
              from the catchment root
        - Shorthand column names are accepted:
            - 'erosion_y1' resolves to 'erosion_y1_sum'
            - 'peak_erosion_y1' resolves to 'peak_erosion_y1_mean'
            - 'mass' resolves to const.DEBRIS_MASS_FIELD
        - Positional calling convention (catchment, data_folder,
          colour_col) is also accepted for backwards compatibility,
          e.g. plot_subcatchments(name, 'DebrisFlow', 'Year1_num_events')
        ----------------------------------------------------------------
        """
        # Support legacy positional calling convention
        # (catchment, folder_name, colour_col). If colour_col looks
        # like a folder name and data_type has been provided, swap:
        _known_folders = {
            'DebrisFlow', 'Results', 'Topography', 'Soils',
            'Erodibility', 'Delivery', 'Subcatchments',
            }
        if colour_col in _known_folders and data_type is not None:
            colour_col, data_type = data_type, colour_col

        # Auto-detect data_file from colour_col when not supplied;
        # also set data_type if not already provided:
        if data_file is None and table is None:
            col_lower = colour_col.lower()
            if any(k in col_lower for k in ('erosion', 'delivered')):
                if data_type is None:
                    data_type = const.RESULTS_FOLDER_NAME
                data_file = const.RUSLE_SC_SUMMARY_NAME
            elif any(
                k in col_lower
                for k in ('events', 'debris', 'mass', 'i12')
                ):
                if data_type is None:
                    data_type = 'DebrisFlow'
                data_file = const.DEBRIS_SC_SUMMARY_NAME
            else:
                if data_type is None:
                    data_type = ''
                data_file = 'Soil_Slope_Aridity_dNBR_subcatchments'

        subcatch_gdf = self.get_subcatchments(catchment)

        non_geo_data = self.get_table_safely(
            colour_col=colour_col,
            data_type=data_type,
            data_file=data_file,
            catchment=catchment,
            allow_basic=True,
            table=table,
            )

        # Resolve shorthand column names before looking up the data:
        if non_geo_data is not None:
            # 'mass' -> the actual debris mass delivery column:
            if colour_col.lower().strip() == 'mass':
                colour_col = const.DEBRIS_MASS_FIELD

            # Bare column names -> append the default aggregation
            # suffix. Rules:
            #   i12 columns  -> _min (most vulnerable headwater)
            #   peak rasters -> _mean
            #   total rasters -> _sum
            if colour_col not in non_geo_data.columns:
                _cl = colour_col.lower()
                if 'i12' in _cl:
                    suffix = '_min'
                elif 'peak' in _cl or 'max' in _cl:
                    suffix = '_mean'
                else:
                    suffix = '_sum'
                candidate = colour_col + suffix
                if candidate in non_geo_data.columns:
                    logger.info(
                        f'Column {colour_col} not found; '
                        f'resolving to {candidate}.'
                        )
                    colour_col = candidate

        id_col = self.subcatchment_id
        if non_geo_data is not None:
            ng_for_join = non_geo_data[[id_col, colour_col]]
        else:
            ng_for_join = None

        vis_params = self.get_vis_params(colour_col)

        # Copy before modifying - vis_params dicts are shared instance
        # attributes and must not be mutated in-place:
        vis_params = vis_params.copy()

        # Set title and units for erosion/delivery columns. The
        # aggregation suffix on the column name (_sum or _mean) tells
        # us exactly what was computed, so we can label it precisely:
        col_lower = colour_col.lower()
        if 'erosion' in col_lower or 'delivered' in col_lower:
            var_type = (
                'Erosion' if 'erosion' in col_lower else 'Delivered'
                )
            year = (
                'Year 1' if 'y1' in col_lower
                else 'Year 2' if 'y2' in col_lower
                else ''
                )
            if colour_col.endswith('_mean'):
                # Peak rasters: each cell stores the max 30-min value.
                # Zonal stat is mean across cells - 'mean tonnes per
                # cell' distinguishes it from a catchment total:
                agg = 'Peak 30-min'
                vis_params['units'] = 'mean peak tonnes per cell'
            else:
                # Total rasters: each cell stores cumulative tonnes.
                # Zonal stat is a sum - total tonnes eroded within
                # the subcatchment:
                agg = 'Total'
                vis_params['units'] = 'total tonnes'
            vis_params['title_varname'] = (
                f'{agg} {var_type} {year}'.strip()
                )

        elif 'i12' in col_lower:
            # I12 threshold columns: suffix tells us min or mean.
            # Year is encoded as 'year_1' / 'year_2' in the col name:
            year = (
                'Year 1' if 'year_1' in col_lower
                else 'Year 2' if 'year_2' in col_lower
                else ''
                )
            stat_desc = (
                'Min' if col_lower.endswith('_min') else 'Mean'
                )
            vis_params['title_varname'] = (
                f'{stat_desc} I12 Threshold {year}'.strip()
                )
            # Units are already 'mm/hr' from vis_i12_crit - correct.

        # Build the axes title. When title_varname is fully specified,
        # use it directly - make_axes_title sniffs year/agg from the
        # column name and would duplicate them for suffixed columns
        # like 'peak_erosion_y1_mean'. Fall back to make_axes_title
        # only for columns with no recognised title_varname:
        if vis_params.get('title_varname'):
            catch_label = toputil.clean_chart_title(catchment)
            ax_title = (
                f'{catch_label} Subcatchments: '
                f'{vis_params["title_varname"]}'
                )
        else:
            ax_title = toputil.make_axes_title(
                catchment,
                'Subcatchments',
                vis_params['title_varname'],
                colour_col,
                )

        self.plot_catchment_polygons(
            catchment=catchment,
            polygons=subcatch_gdf,
            colour_col=colour_col,
            vis_params=vis_params,
            title=ax_title,
            non_geo_data=ng_for_join,
            id_col=id_col,
            existing_figure=existing_figure,
            existing_axes=existing_axes,
            )

    # --- Data access --------------------------------------------------------

    ###########################################################################
    def get_saved_data(
        self,
        catchment: str,
        type: str | None,
        name: str,
        format: str = 'csv',
        ) -> pd.DataFrame:
        """
        Read a file saved within a catchment's folder structure.

        Parameters:
        - catchment: Name of the catchment.
        - type: Subfolder name (e.g. 'DebrisFlow'), or None to read
          from the catchment root.
        - name: File name without extension.
        - format: File extension (default 'csv').

        Returns:
        - DataFrame read from the specified file.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        if type is None:
            data_table_loc = self.catchment_path(catchment)
        else:
            data_table_loc = self.catchment_path(catchment, type)
        data_table_path = (
            os.path.join(data_table_loc, name) + '.' + format
            )
        df = pd.read_csv(data_table_path)
        return df

    ###########################################################################
    def get_table_safely(
        self,
        colour_col: str,
        data_type: str,
        data_file: str,
        catchment: str,
        allow_basic: bool,
        table: pd.DataFrame | None = None,
        ):
        """
        Load a non-spatial table with basic sanity checks for polygon
        plotting.

        Parameters:
        - colour_col: Column name to be used for colouring polygons.
        - data_type: Subfolder name within the catchment directory.
        - data_file: CSV file name without extension.
        - catchment: Name of the catchment.
        - allow_basic: If True, return None (rather than raising)
          when the file is not found, so polygons are still plotted
          in a uniform colour.
        - table: Optional pre-loaded DataFrame. Skips file loading
          if provided.

        Returns:
        - DataFrame if loaded or provided, or None if the file was
          not found and allow_basic is True.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        if table is None:
            try:
                non_geo_data = self.get_saved_data(
                    catchment=catchment,
                    type=data_type,
                    name=data_file,
                    )
            except FileNotFoundError:
                if allow_basic:
                    logger.info(
                        'Plotting polygons was requested with no '
                        'data to colour the shapes with. Proceeding '
                        'with uniform colours.'
                        )
                    non_geo_data = None
                else:
                    raise
        else:
            non_geo_data = table

        # Warn if the required column is missing from the loaded data:
        if non_geo_data is not None:
            if colour_col not in non_geo_data.columns:
                logger.warning(
                    f'project.plot_subcatchments() was asked to '
                    f'colour the map based on {colour_col}, but the '
                    f'data table only had: '
                    f'{list(non_geo_data.columns)}. Plotting will '
                    f'proceed with uniform colours.'
                    )
        return non_geo_data

    ###########################################################################
    def thresh_sev_scatter(
        self,
        catchment: str,
        existing_figure=None,
        existing_axes=None,
        width=12,
        height=8,
        dpi=600,
        ):
        """
        Scatter plot of year 1 and year 2 critical rainfall intensity
        thresholds vs. mean dNBR for each headwater.

        Parameters:
        - catchment: Name of the catchment.
        - existing_figure: matplotlib figure to plot onto.
        - existing_axes: matplotlib axes to plot onto.
        - width: Figure width in inches.
        - height: Figure height in inches.
        - dpi: Figure resolution in dots per inch.

        Returns:
        - matplotlib figure object.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        # Load the DebrisFlow data for this catchment:
        folder = self.catchment_path(catchment, 'DebrisFlow')
        file = 'DebrisFlowData.csv'
        path = os.path.join(folder, file)

        if not os.path.isfile(path):
            raise FileNotFoundError(
                'project.thresh_sev_scatter() requires debris flow '
                'data to be loaded. Run debris.debris_flow() which '
                f'will save the required data here:\n{path}'
                )
        non_geo_data = pd.read_csv(path)

        # Extract the threshold and dNBR columns, dropping any rows
        # with missing values:
        x1_col = 'I12_crit_mean_Year_1'
        x2_col = 'I12_crit_mean_Year_2'
        y_col = 'dNBR_mean'
        data_for_scatter = (
            non_geo_data[[x1_col, x2_col, y_col]].dropna()
            )

        # Compute median values to use as reference lines on the plot:
        median_x1_col = data_for_scatter[x1_col].median()
        median_x2_col = data_for_scatter[x2_col].median()
        median_y_col = data_for_scatter[y_col].median()

        col_year_1 = '#800080'  # purple
        col_year_2 = '#696969'  # grey

        # Set up figure size and resolution:
        sfig, sax = toputil.fig_ax_admin(existing_figure, existing_axes)
        sfig.set_size_inches(width, height)
        sfig.set_dpi(dpi)

        # Plot year 1 and year 2 threshold values as separate scatter
        # series so they can be distinguished by marker and colour:
        sax.scatter(
            x=data_for_scatter[x1_col],
            y=data_for_scatter[y_col],
            marker='x',
            color=col_year_1,
            label='Year 1',
            )
        sax.scatter(
            x=data_for_scatter[x2_col],
            y=data_for_scatter[y_col],
            marker='o',
            color=col_year_2,
            label='Year 2',
            )
        sax.set_ylim(0, 1000)

        # Vertical median lines for each year's critical threshold:
        sax.axvline(
            x=median_x1_col,
            label='I12 crit. rain threshold: y1 median',
            ls='--',
            c=col_year_1,
            )
        sax.axvline(
            x=median_x2_col,
            label='I12 crit. rain threshold: y2 median',
            ls='--',
            c=col_year_2,
            )
        # Horizontal line for the dNBR median:
        sax.axhline(
            y=median_y_col, label='dNBR median', ls=':', c='grey'
            )

        # Add title, axis labels, and legend:
        catch_title = toputil.clean_chart_title(catchment)
        sax.set_title(
            'Scatter plot of mean dNBR vs year 1 critical rainfall '
            f'for {catch_title} headwaters'
            )
        sax.set_xlabel('I12 critical threshold for debris flow')
        sax.set_ylabel('Mean dNBR')
        sax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

        return sfig

    # --- Fire metadata ------------------------------------------------------

    ###########################################################################
    def get_fire_end_date(self, catchment):
        """
        Get the fire end date for a catchment as a pandas Timestamp.

        Parameters:
        - catchment: Name of the catchment.

        Returns:
        - pandas Timestamp of the fire end date.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        fire_meta_path = self.catchment_path(
            catchment,
            const.FIRE_SEVERITY_FOLDER_NAME,
            'FireMeta.csv',
            )
        fire_meta = pd.read_csv(fire_meta_path, index_col=0)
        end_date_iso = fire_meta.loc['end_date', 'Value']
        return pd.to_datetime(end_date_iso)

    # --- Event run-context (fire dates + recovery breakpoints) --------------

    def _run_context_path(self, catchment, *, event=None):
        """
        Return the path to a catchment's run-context file.

        The ``event`` keyword is reserved for the multi-event model, where
        the run-context is scoped per event rather than per catchment; it
        is currently ignored.
        """
        return self.catchment_path(catchment, const.RUN_CONTEXT_NAME)

    def get_run_context(self, catchment, *, event=None) -> EventRunContext:
        """
        Load a catchment's event run-context.

        Falls back to a reconstructed context (fire dates from FireMeta.csv
        if present, default recovery breakpoints) when no RunContext.json
        exists yet, logging a warning that recommends re-running
        compute_adjusted_k_c to persist a proper context.

        Parameters:
        - catchment: Name of the catchment.
        - event: Reserved for multi-event scoping (currently ignored).

        Returns:
        - EventRunContext for the catchment.
        """
        path = self._run_context_path(catchment, event=event)
        if os.path.exists(path):
            with open(path) as f:
                return EventRunContext.from_dict(json.load(f))

        # Legacy fallback: no run-context persisted yet.
        fire_start = fire_end = None
        try:
            fire_meta_path = self.catchment_path(
                catchment, const.FIRE_SEVERITY_FOLDER_NAME, 'FireMeta.csv')
            fire_meta = pd.read_csv(fire_meta_path, index_col=0)
            fire_start = pd.to_datetime(fire_meta.loc['start_date', 'Value'])
            fire_end = pd.to_datetime(fire_meta.loc['end_date', 'Value'])
        except (FileNotFoundError, KeyError):
            pass
        logger.warning(
            'No %s for catchment %s; using a fallback run-context (fire '
            'dates from FireMeta.csv if present, default recovery '
            'breakpoints). Re-run compute_adjusted_k_c to persist one.',
            const.RUN_CONTEXT_NAME, catchment,
        )
        return EventRunContext(
            fire_start_date=fire_start,
            fire_end_date=fire_end,
            recovery_breakpoints=list(const.DEFAULT_RECOVERY_BREAKPOINTS),
        )

    def set_run_context(self, catchment, ctx: EventRunContext, *, event=None):
        """
        Write a catchment's event run-context to RunContext.json.

        Parameters:
        - catchment: Name of the catchment.
        - ctx: EventRunContext to persist.
        - event: Reserved for multi-event scoping (currently ignored).
        """
        path = self._run_context_path(catchment, event=event)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(ctx.to_dict(), f, indent=2)
        return ctx

    def update_run_context(self, catchment, *, event=None, **fields):
        """
        Update selected fields of a catchment's run-context and persist it.

        Reads the current (or fallback) context, replaces the given fields,
        and writes the result. Accepts fire_start_date, fire_end_date, and
        recovery_breakpoints.

        Returns the updated EventRunContext.
        """
        from dataclasses import replace
        ctx = self.get_run_context(catchment, event=event)
        updated = replace(ctx, **fields)
        return self.set_run_context(catchment, updated, event=event)

    def get_simulation_period(
        self, catchment, *, fire_end_date=None, event=None
    ):
        """
        Return the (start, end) pandas Timestamps of the recovery
        simulation period for a catchment.

        The period spans the recovery windows recorded in the run-context —
        from the fire end date through the end of the last window — so the
        simulation-period end never has to be hard-coded. Pass fire_end_date
        to override the stored value (e.g. when the notebook drives the fire
        date explicitly).

        Parameters:
        - catchment: Name of the catchment.
        - fire_end_date: Optional fire end date override.
        - event: Reserved for multi-event scoping (currently ignored).

        Returns:
        - (start, end) tuple of pandas Timestamps, suitable for
          get_rainfall_replicates and aggregate_rainfall_data.
        """
        ctx = self.get_run_context(catchment, event=event)
        if fire_end_date is not None:
            from dataclasses import replace
            ctx = replace(ctx, fire_end_date=pd.Timestamp(fire_end_date))
        return ctx.simulation_period()


###############################################################################
def plot_catchment_boundary(
    project: FireImpactsProject,
    catchment: str,
    axes,
    new_legend=True,
    ):
    """
    Plot the catchment boundary on an axes object.

    Parameters:
    - project: FireImpactsProject instance.
    - catchment: Name of the catchment to plot.
    - axes: matplotlib Axes to plot onto.
    - new_legend: If True, add a legend entry for the boundary line.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    catch_bound_colour = 'red'
    gdf = project.catchment_boundary(catchment)
    gdf.plot(ax=axes, facecolor='none', edgecolor=catch_bound_colour)

    # Dummy line object used only to create a legend entry:
    dummy_line = [
        mlines.Line2D([], [], color=catch_bound_colour)
        ]
    if new_legend:
        axes.legend(
            dummy_line,
            ['Catchment Boundary'],
            fontsize='xx-small',
            )


###############################################################################
def get_vis_dx(ax, crs):
    """
    Return visualisation dx (map units per pixel) for scalebar use.

    Parameters:
    - ax: matplotlib Axes object.
    - crs: Coordinate reference system object.

    Returns:
    - Map units per pixel, or None if the CRS is not projected.
    --------------------------------------------------------------------
    Notes:
    - Probably only required when the extent property has not already
      been set by matplotlib or geopandas.
    --------------------------------------------------------------------
    """
    if not crs.is_projected:
        logger.warning(
            'get_vis_dx only accepts projected CRS objects. '
            'dx not returned.'
            )
        return None
    # Calculate map units per pixel from axis limits and pixel width:
    x_range = ax.get_xlim()
    ax_width_map_units = x_range[1] - x_range[0]

    ax_bbox_pixels = ax.get_window_extent()
    ax_width_px = ax_bbox_pixels.width

    map_units_per_pixel = ax_width_map_units / ax_width_px
    return map_units_per_pixel


###############################################################################
def find_all_shapefiles(base_directory):
    """
    Find all shapefiles in a directory tree.

    Parameters:
    - base_directory: Root directory to search recursively.

    Returns:
    - List of absolute paths to all .shp files found.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    assert os.path.isdir(base_directory), (
        f'Directory not found: {base_directory}'
        )
    shapefiles = glob(
        os.path.join(base_directory, '**', '*.shp'), recursive=True
        )
    return shapefiles


###############################################################################
def _filter_zones_by_masked_dnbr(
    project: FireImpactsProject,
    catchment_name: str,
    zones_gdf: gpd.GeoDataFrame,
    id_col: str,
    masked_nan_threshold: float,
    ) -> gpd.GeoDataFrame:
    """
    Drop zones where the NaN fraction in masked_dNBR exceeds a
    threshold.

    Parameters:
    - project: FireImpactsProject instance.
    - catchment_name: Name of the catchment.
    - zones_gdf: GeoDataFrame of zone polygons (headwaters or
      subcatchments) with an id_col column.
    - id_col: Column identifying each zone.
    - masked_nan_threshold: Maximum NaN fraction (0-1) before a
      zone is excluded.

    Returns:
    - Filtered copy of zones_gdf.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    import rasterio
    from rasterio.features import rasterize

    masked_dnbr_path = project.catchment_path(
        catchment_name, 'FireSeverity', 'masked_dNBR.tif'
        )
    if not os.path.exists(masked_dnbr_path):
        logger.warning(
            f'masked_dNBR.tif not found for {catchment_name} '
            f'- skipping NaN threshold filtering.'
            )
        return zones_gdf

    # Read the masked dNBR raster and reproject zones to match:
    with rasterio.open(masked_dnbr_path) as src:
        dnbr_data = src.read(1)
        dnbr_transform = src.transform
        dnbr_crs = src.crs

    zones_reproj = zones_gdf.to_crs(dnbr_crs)

    # For each zone, rasterize its geometry and count NaN pixels:
    exclude_ids = []
    for idx, zone in zones_reproj.iterrows():
        zone_mask = rasterize(
            [zone.geometry],
            out_shape=dnbr_data.shape,
            transform=dnbr_transform,
            fill=0,
            default_value=1,
            dtype=np.uint8,
            )
        inside = zone_mask == 1
        n_pixels = int(inside.sum())
        if n_pixels == 0:
            continue
        n_nan = int(np.isnan(dnbr_data[inside]).sum())
        nan_frac = n_nan / n_pixels
        if nan_frac > masked_nan_threshold:
            zone_id = zone[id_col]
            exclude_ids.append(zone_id)
            logger.info(
                f'Excluding {id_col} {zone_id} from summary stats: '
                f'{nan_frac * 100:.1f}% of pixels are NaN in masked '
                f'dNBR (threshold {masked_nan_threshold * 100:.1f}%)'
                )

    # Remove the flagged zones and report how many were dropped:
    if exclude_ids:
        n_before = len(zones_gdf)
        zones_gdf = zones_gdf[
            ~zones_gdf[id_col].isin(exclude_ids)
            ].copy()
        logger.info(
            f'Excluded {len(exclude_ids)} of {n_before} zones '
            f'exceeding {masked_nan_threshold * 100:.0f}% NaN '
            f'threshold in masked dNBR for {catchment_name}.'
            )

    return zones_gdf


###############################################################################
def summary_stats(
    project: FireImpactsProject,
    catchment_name=None,
    zone_type='headwaters',
    masked_nan_threshold: float = 0.05,
    layer_nan_threshold: float = 0.05,
    save_shp=False,
    ):
    """
    Calculate summary statistics for a catchment from pre-processed
    raster data.

    Parameters:
    - project: FireImpactsProject instance, or a path string from
      which to load one.
    - catchment_name: Name of the catchment to process. If not
      provided, processes all catchments in the project.
    - zone_type: 'headwaters' or 'subcatchments'.
    - masked_nan_threshold: For headwaters only - maximum fraction
      of a headwater's area that may be NaN in masked_dNBR.tif
      before the headwater is excluded. Default 0.05 (5%).
      Headwaters exceeding this (e.g. containing a lake) are
      dropped.
    - layer_nan_threshold: Per-layer, per-zone threshold for
      missing data. When a zone has more than this fraction of its
      overlapping pixels as nodata in a particular raster layer,
      all statistics for that zone/layer combination are set to
      NaN. For coarse rasters, all_touched=True is used
      automatically. Default 0.05 (5%).
    - save_shp: If True, also save results as a shapefile alongside
      the CSV.

    Returns:
    - pd.DataFrame of summary statistics for the catchment, or a
      dict of DataFrames if catchment_name was not provided.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    acceptable_zones = ['headwaters', 'subcatchments']
    requested_zone = zone_type.strip().lower()
    if requested_zone not in acceptable_zones:
        raise ValueError(
            'project.summary_stats() was asked to compute stats '
            f'for {zone_type}. Please use one of: {acceptable_zones}'
            )

    # If given a path string, load a project from that path:
    if isinstance(project, str):
        project = FireImpactsProject(project)

    # Process for all catchments if none was specified:
    if catchment_name is None:
        return project.for_each_catchment(
            lambda c: summary_stats(
                project, c,
                zone_type=zone_type,
                masked_nan_threshold=masked_nan_threshold,
                layer_nan_threshold=layer_nan_threshold,
                save_shp=save_shp,
                )
            )

    # Load the appropriate zone polygons and optionally filter them:
    if requested_zone == 'subcatchments':
        id_col_name = project.subcatchment_id
        zones_gdf = project.get_subcatchments(catchment_name)
    else:
        id_col_name = project.headwater_id
        zones_gdf = project.get_headwaters(catchment_name)

        # Filter out headwaters where too much area is NaN in the
        # masked dNBR grid (e.g. water bodies, non-vegetation):
        logger.info(
            f'Filtering {zone_type} in {catchment_name} based on '
            f'NaN fraction in masked dNBR...'
            )
        zones_gdf = _filter_zones_by_masked_dnbr(
            project, catchment_name, zones_gdf,
            id_col_name, masked_nan_threshold,
            )

    # Build the list of raster layers to extract stats from.
    # Fixed layers first; then discover all .tif files under Soils/:
    sources = [
        ('Slope', ('Topography', 'Slope.tif')),
        ('dNBR',  ('FireSeverity', 'dNBR.tif')),
        ('Aridity', ('Soils', 'Aridity.tif')),
        # ('Rain', 'Rain', 'Rainfall.tif')
        ]

    soil_path = project.catchment_path(catchment_name, 'Soils')
    for fn in os.listdir(soil_path):
        abs_fn = os.path.join(soil_path, fn)
        if not os.path.isdir(abs_fn):
            continue
        for child_fn in os.listdir(abs_fn):
            if child_fn.endswith('.tif'):
                sources.append(
                    (
                        child_fn.replace('.tif', ''),
                        ('Soils', fn, child_fn),
                        )
                    )

    # Reset index after filtering so that list-based columns from
    # rasterstats align correctly with the id column:
    zones_gdf = zones_gdf.reset_index(drop=True)

    result = {id_col_name: zones_gdf[id_col_name].tolist()}

    # Determine the reference resolution from the DEM so we can
    # detect coarser layers and use all_touched for them:
    dem_path = project.catchment_path(
        catchment_name, 'Topography', 'DEM.tif'
        )
    with rio.open(dem_path) as dem_src:
        ref_res = dem_src.res[0]

    logger.info(
        f'Processing {len(zones_gdf)} polygons for '
        f'{len(sources)} layers in {catchment_name}'
        )
    for label, path in sources:
        logger.info(f'Processing {label} from {path[-1]}')
        raster_path = project.catchment_path(catchment_name, *path)

        # Use all_touched for rasters coarser than 2x the DEM
        # resolution, so small zones still capture pixels:
        with rio.open(raster_path) as layer_src:
            layer_res = layer_src.res[0]
        use_all_touched = layer_res > ref_res * 2
        if use_all_touched:
            logger.info(
                f'{label}: resolution {layer_res:.0f}m is coarser '
                f'than DEM ({ref_res:.0f}m) - using all_touched=True'
                )

        stats = toputil.get_zonal_stats(
            zones_gdf,
            raster_path,
            label,
            extra_stats=['count', 'nodata'],
            all_touched=use_all_touched,
            )

        # Detect zones where stats could not be computed.
        #
        # We cannot rely on rasterstats' 'nodata' count for NaN-nodata
        # rasters (nan == nan is False, so the count is always 0).
        # Instead, check the output stats directly: if rasterstats
        # returned None for any core stat, the zone had no usable
        # pixels - either zero overlap or all pixels were nodata/NaN.
        #
        # For zones that DO have valid stats, apply the
        # layer_nan_threshold using the count vs total pixel estimate.
        zones_no_data = 0
        zones_nulled = 0

        assert len(stats) == len(zones_gdf), (
            'Length of stats does not match number of zones. '
            'Expected %d, got %d.' % (len(zones_gdf), len(stats))
            )

        for s in stats:
            # When all pixels are nodata/NaN or there is zero overlap,
            # rasterstats returns None for stats like 'mean':
            sample_stat = s.get('mean')
            if sample_stat is None:
                for k in STATS:
                    s[k] = np.nan
                zones_no_data += 1
                continue

            # For zones with valid stats, check the nodata fraction.
            # Use 'count' (valid pixels) vs total rasterized pixels.
            # Note: 'nodata' count from rasterstats is unreliable for
            # NaN-nodata rasters, so we estimate total from count +
            # nodata, falling back to count-only if nodata is 0.
            n_valid = s.get('count', 0) or 0
            n_nodata = s.get('nodata', 0) or 0
            n_total = n_valid + n_nodata
            if n_total > 0 and n_nodata > 0:
                if (n_nodata / n_total) > layer_nan_threshold:
                    for k in STATS:
                        s[k] = np.nan
                    zones_nulled += 1

        if zones_no_data > 0:
            logger.warning(
                f'{label}: {zones_no_data} of {len(stats)} zones '
                f'returned no valid statistics (no pixel overlap '
                f'or all pixels are nodata/NaN, possibly due to '
                f'coarse raster resolution).'
                )
        if zones_nulled > 0:
            logger.warning(
                f'{label}: {zones_nulled} of {len(stats)} zones '
                f'exceed {layer_nan_threshold * 100:.0f}% nodata '
                f'threshold - stats set to NaN for those zones.'
                )

        for k in STATS:
            # rasterstats returns Python None (not np.nan) for zones
            # that fall entirely within nodata pixels. Coerce to nan
            # so the column stays float64 rather than object dtype.
            # Object dtype causes to_csv() to write values as strings
            # which then can't be reliably read back as numbers.
            result[f'{label}_{k}'] = [
                float('nan') if s[k] is None else s[k]
                for s in stats
                ]

    extracted_data = pd.DataFrame(result)
    extracted_data = extracted_data.apply(
        pd.to_numeric, errors='coerce'
        )

    # Preserve raw (pre-clip) dNBR stat columns for diagnostics.
    # These sit alongside the standardised columns in the output:
    for stat in STATS:
        raw_col = f'dNBR_{stat}'
        extracted_data[f'dNBR_raw_{stat}'] = extracted_data[raw_col]

    # Convert dNBR stats to standardised values [0, 1000]: clip
    # negatives to 0 then scale. Note this clips the already-
    # aggregated stats (e.g. a negative mean clips to 0), which can
    # make mean=0 while max>0. The raw columns let you verify the
    # pre-clip values if the result looks unexpected.
    for stat in STATS:
        col = f'dNBR_{stat}'
        extracted_data[col] = format_dNBR(extracted_data[col])

    # -----------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------
    base_name = f'Soil_Slope_Aridity_dNBR_{zone_type}'
    csv_path = project.catchment_path(
        catchment_name, f'{base_name}.csv'
        )
    extracted_data.to_csv(csv_path, index=False)
    logger.info(f'[write] {csv_path}')

    if save_shp:
        # Join computed stats back onto zone geometries:
        shp_gdf = zones_gdf[[id_col_name, 'geometry']].merge(
            extracted_data, on=id_col_name, how='left'
            )
        shp_path = project.catchment_path(
            catchment_name, f'{base_name}.shp'
            )
        shp_gdf.to_file(shp_path)
        logger.info(f'[write] {shp_path}')

    return extracted_data


###############################################################################
def format_dNBR(series: pd.Series):
    """
    Convert dNBR values to a standardised [0, 1000] range.

    Parameters:
    - series: pandas Series of dNBR values in the range -1 to 1.

    Returns:
    - Series with values clipped to 0 at the lower end and scaled
      to the 0-1000 range.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    return series.clip(lower=0).mul(1000).astype(np.float64)


###############################################################################
def save_catchment_raster(
    project: FireImpactsProject,
    catchment_name: str,
    file_name: str,
    section: str,
    data,
    meta,
    ):
    """
    Write a raster array to a catchment's folder structure.

    Parameters:
    - project: FireImpactsProject instance.
    - catchment_name: Name of the catchment.
    - file_name: Output file name without extension.
    - section: Sub-folder within the catchment directory
      (e.g. 'Erodibility').
    - data: Numpy array to write.
    - meta: Rasterio metadata dict for the output raster.

    Returns:
    - Tuple of (success: bool, message: str).
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    out_path = project.catchment_path(
        catchment_name, section, f'{file_name}.tif'
        )

    # Standardise the output dtype based on the input array kind:
    final_meta = meta.copy()
    in_dtype = final_meta['dtype']
    in_dtype_kind = np.dtype(in_dtype).kind
    out_dtype = default_dtypes_raster[
        numpy_kind_to_desc[in_dtype_kind]
        ]
    final_meta.update(dtype=out_dtype, count=1)

    try:
        with rio.open(out_path, 'w', **final_meta) as dst:
            dst.write(data.astype(out_dtype), 1)
        result = True
        result_string = f'Saved raster to {out_path}'
    except Exception as e:
        result = False
        result_string = f'Could not save raster: {e}'

    return result, result_string
