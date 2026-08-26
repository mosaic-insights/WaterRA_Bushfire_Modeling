"""
Shared constants, conversion factors, file names, and field names
used throughout the fire_impacts package.
"""

# ------- Conversions: -------------------------------------------------
# Basic units:
M_TO_KM = 1e-3
M2_TO_HA = 1e-4
PERCENT_TO_FRACTION = 1e-2
MILLIGRAMS_TO_KILOGRAMS = 1e-6
APPROX_KM_PER_DEGREE = 111
APPROX_DEGREES_TO_METRES = 111000
KG_TO_TONNES = 1e-3

# ------- General lookups: ---------------------------------------------
CRS_METRE_UNITS = {'m', 'meter', 'meters', 'metre', 'metres'}
STATS = ['mean', 'max', 'min', 'median', 'std']

# ------- Hydrology constants: -----------------------------------------
DEFAULT_HW_THRESHOLD = 20000
D8_FLOW_DIRECTIONS = (
    64,   # north
    128,  # northeast
    1,    # east
    2,    # southeast
    4,    # south
    8,    # southwest
    16,   # west
    32,   # northwest
    )
FLOW_ROUTING_TYPE = 'd8'

# ------- Debris flow constants: ---------------------------------------
HILLSLOPE_AREA = 1.3e4
CHANNELISED_FLOW_THRESHOLD = 1.4e7
SEDIMENT_BULK_DENSITY = 1270  # kg/m3
ROCK_BULK_DENSITY = 2220  # kg/m3
HILLSLOPE_ROCK_FRACTION = 0.12
CHANNEL_ROCK_FRACTION = 0.45
HILLSLOPE_PARAMETERS = dict(
    ae=4.5e-4,
    be=0.36,
    ad=0.3 * 4.5e-4,
    bd=0.36,
    rock=HILLSLOPE_ROCK_FRACTION
    )
CHANNEL_PARAMETERS = dict(
    ae=4.1e-4,
    be=0.52,
    ad=3.7e-7,
    bd=1.06,
    rock=CHANNEL_ROCK_FRACTION
    )
NUM_SIM_YEARS = 2

# Length of a debris-flow simulation year, in days.
#
# Deliberately 365, not 365.25. The two are not interchangeable here and
# neither is unambiguously right: 365 is exact for three years in four,
# while 365.25 is the better long-run average — and which applies depends
# on whether the driving rainfall contains leap days at all. Stochastic
# replicates (pyraingen) may not; historical series do. See
# issues/debris-flow-year-length.md — this is parked, not settled.
DAYS_PER_SIM_YEAR = 365

# The I12 lookup tabulates thresholds at discrete times since fire — for
# the packaged table, 0.434 and 1.434 years (roughly 5 and 17 months).
# These are treated as REPRESENTATIVE OF THEIR WHOLE YEAR: the threshold
# fitted at 0.434 years is applied across the entire first year after the
# fire, and the one at 1.434 across the second. The model is therefore
# piecewise-constant in time since fire, with one step per year, rather
# than varying continuously. A table with more `years` bins would give
# more steps; the code reads the bins rather than assuming two.

# ------- Rainfall kinetic energy: --------------------------------------
# Unit kinetic energy follows the exponential KE-intensity form
#
#     e_r = 0.29 * [1 - 0.72 * exp(-k * i_r)]
#
# with e_r in MJ/ha/mm and i_r in mm/h. 0.29 is the asymptotic maximum
# (drops reach terminal velocity, so energy per mm saturates) and 0.72
# fixes the drizzle floor at 0.29*(1-0.72) = 0.0812 MJ/ha/mm. Both are
# fixed by the equation's form; only the rate constant k varies between
# published versions:
#
#   0.05   Brown & Foster (1987), as used in RUSLE (Renard et al. 1997).
#   0.082  McGregor et al. (1995), adopted by USDA-ARS (2013) in RUSLE2.
#
# We use the RUSLE2 value. It is the one supported by Australian
# evidence: RUSLE's 0.05 was found to underestimate unit energy here
# (Yu 1999), and the exponential form itself derives from Rosewell
# (1986), measured in eastern Australia.
#
# Changing k selects a model version rather than tuning one, so it is not
# a free parameter — but the literature does report regional KE-I
# relationships, so it stays exposed as
# params.ErosionParams.kinetic_energy_coefficient. Reference:
# Yin, Nearing, Borrelli & Xue (2017), "Rainfall Erosivity: An Overview
# of Methodologies and Applications", Vadose Zone Journal, eqs. [2], [3].
KE_ASYMPTOTE = 0.29           # MJ/ha/mm
KE_FLOOR_FRACTION = 0.72
DEFAULT_KE_RATE_RUSLE2 = 0.082    # McGregor et al. (1995) / RUSLE2
DEFAULT_KE_RATE_RUSLE = 0.05      # Brown & Foster (1987) / RUSLE

# ------- dNBR scale: ---------------------------------------------------
# dNBR is *stored* as the raw band-ratio difference (pre-fire NBR minus
# post-fire NBR, negatives clipped), which lands in roughly [0, 1]. It is
# *quoted and thresholded* on the conventional 0-1000 scale used
# throughout the fire-severity literature, and so are every threshold and
# lookup table in this package.
#
# DNBR_SCALE converts stored -> conventional. Read dNBR through
# pre.util.read_dnbr_* rather than applying it by hand: consumers
# previously each remembered (or forgot) to multiply, and one that forgot
# compared a [0, 1] raster against a 400 threshold, so the whole
# high-severity branch was unreachable.
#
# Every threshold below, and params.FireAdjustmentParams.dnbr_saturation /
# params.ErosionParams.dnbr_severity_threshold, are on the conventional
# scale. Keep them here beside the factor: a threshold that drifts onto
# the other scale is silent, not loud.
DNBR_SCALE = 1000

# Cells at or above this are "high severity" when splitting erosion
# outputs for reporting (params.ErosionParams.dnbr_severity_threshold).
DEFAULT_DNBR_SEVERITY_THRESHOLD = 400

# dNBR at which the fire-adjusted C factor saturates at its peak value
# (params.FireAdjustmentParams.dnbr_saturation).
DEFAULT_DNBR_SATURATION = 400

# Default hydrogeomorphic-hazard lookup: critical 12-minute rainfall
# intensity keyed on (aridity, dNBR, years since fire, slope gradient).
# The filename encodes the fitted coefficient b = 30.27. This table *is*
# the debris-flow triggering model, so it is overridable — but no tooling
# is provided to build an alternative, and the default is expected to
# stand for the foreseeable future.
DEFAULT_I12_LOOKUP = 'HFlookup_b30pt27.csv'

# Headwaters with a mean dNBR below this value are excluded from the
# debris-flow analysis (they are considered insufficiently burnt).
# Compared against the dNBR_mean column of summary_stats, which is on the
# conventional scale.
DEFAULT_DEBRIS_DNBR_THRESHOLD = 100

# ------- Fire recovery time and intervals: ---------------------------------------------
# Recovery is specified as a single monotonic array of *breakpoints* in
# years since the fire end date. n+1 breakpoints define n contiguous
# recovery windows: window i spans [b_i, b_{i+1}) and is modelled at
# recovery time b_i (the window start). This replaces the old
# (recovery_times + interval) pair, which was redundant for contiguous
# windows and could silently leave gaps/overlaps.
DEFAULT_RECOVERY_BREAKPOINTS = [0, 0.5, 1, 1.5, 2, 2.5, 3]

# Deprecated: retained for one release, derived from the breakpoints.
# Prefer DEFAULT_RECOVERY_BREAKPOINTS.
DEFAULT_RECOVERY_TIMES = DEFAULT_RECOVERY_BREAKPOINTS[:-1]
DEFAULT_RECOVERY_INTERVAL_YEARS = 0.5

# Per-event definition file, written at Events/<event>/event.json.
EVENT_DEFINITION_NAME = 'event.json'

# Project-scope calibration parameter overrides, written at
# <project>/parameters.json. Hand-editable and user-owned: deliberately kept
# out of settings.json, which is machine-written (_write() rewrites it
# wholesale whenever a catchment is added) and would drop the key.
PARAMETERS_FILE_NAME = 'parameters.json'

# Resolved parameter provenance record — the values a step actually used,
# where each came from, and a digest. Deliberately a different name from
# parameters.json: that file is the sparse *override input* a user edits,
# this one is the full *resolved output* the library writes. Conflating them
# would turn every package default into an explicit user setting on first
# run, destroying the default/chosen distinction the record exists for.
#
# Written at whichever scope the step produced outputs for:
#   Catchments/<c>/provenance.json
#   Catchments/<c>/Events/<event>/provenance.json
#   Catchments/<c>/Runs/<event>/<ensemble>/<section>/provenance.json
PROVENANCE_FILE_NAME = 'provenance.json'


def recovery_time_suffix(recovery_time: float) -> str:
    """
    Convert a recovery time value into a safe filename suffix.

    Whole numbers normalise to their integer form, so that a breakpoint
    list of ints and one of floats name the same files. The layers are
    written from the breakpoints passed to compute_adjusted_k_c but read
    back from the persisted run-context, and without this a 0 vs 0.0
    mismatch between the two looks like a missing layer.

    Numpy scalars are accepted and normalise the same way.

    Examples
    --------
    0    -> t0
    0.0  -> t0
    0.5  -> t0_5
    1    -> t1
    1.5  -> t1_5
    2.5  -> t2_5
    """
    value = float(recovery_time)
    if value.is_integer():
        value = int(value)
    return f"t{str(value).replace('.', '_')}"


def recovery_windows(breakpoints):
    """
    Convert recovery breakpoints into (start, end) window pairs in years.

    n+1 monotonically increasing breakpoints yield n contiguous windows;
    window i is [breakpoints[i], breakpoints[i+1]) and is modelled at
    recovery time breakpoints[i].

    Raises ValueError if fewer than two breakpoints are given or they are
    not strictly increasing.
    """
    bps = list(breakpoints)
    if len(bps) < 2:
        raise ValueError(
            "recovery breakpoints need at least two values (one window); "
            f"got {bps!r}."
        )
    if any(b <= a for a, b in zip(bps, bps[1:])):
        raise ValueError(
            f"recovery breakpoints must be strictly increasing; got {bps!r}."
        )
    return list(zip(bps[:-1], bps[1:]))


def breakpoints_from_times_and_interval(recovery_times, interval):
    """
    Convert the deprecated (recovery_times, interval) pair into breakpoints.

    Appends a trailing boundary (last start + interval) to close the final
    window. Used to keep deprecated call sites working.
    """
    times = list(recovery_times)
    if not times:
        raise ValueError("recovery_times is empty.")
    return times + [times[-1] + interval]


# ------- Sentinels: ---------------------------------------------------

class _Unset:
    """Sentinel for 'this argument was not supplied'.

    Needed wherever a default is a real, meaningful value: the deprecated
    calibration kwargs (max_sdr, ic0, k, threshold_m2, ...) default to the
    same numbers as the ModelParameters defaults, so `if max_sdr == 0.8`
    cannot distinguish "the user asked for 0.8" from "the user said
    nothing". Without this, an explicit value equal to the default is
    silently dropped and a lower resolution layer wins instead.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return 'UNSET'

    def __bool__(self):
        return False


UNSET = _Unset()


# ------- Dtype standards: ---------------------------------------------

# Convert numpy one-character dtype.kind attributes into more
# general descriptors that map into default_dtypes_raster:
numpy_kind_to_desc = {
    'i': 'int',
    'u': 'int',
    'f': 'float'
    }
NODATA_VAL_INT = -9999

# ------- Standardised file names: ------------------------------------
# Key hydrology files:
DEM_FN = 'DEM'
FLOW_ACCUMULATION_FN = 'Flow_accumulation'
FLOW_DIRECTION_FN = 'Flow_direction'
HEADWATERS_FN = 'Headwaters'
SLOPE_FN = 'Slope'
SLOPE_HYDRO_FN = 'Slope_hydro_enforced'
# Intermediate erosion rasters:
ERO_CUM_M_ALL_FN = 'Erosion_cum_mass_all'
ERO_CUM_M_CLY_FN = 'Erosion_cum_mass_clay'
ERO_CUM_M_SED_FN = 'Erosion_cum_mass_sediment'

# ------- Standardised field names: -----------------------------------
# Debris flow:
CLY_M_ACC_KG = 'Clay mass accumulation (kg)'
TOT_EM_ACC_KG = 'Total erosion mass accumulation (kg)'
TOT_EM_ACC_KG_HA = 'Total erosion mass accumulation (kg/ha)'
SED_M_ACC_KG = 'Sediment mass accumulation (kg)'
DEBRIS_MASS_FIELD = CLY_M_ACC_KG
CATCH_TOTAL_DEBRIS_TONNES = 'Total debris flow mass (tonnes)'
# Headwaters:
HW_ID = 'hw_ID'
HW_ENDP_X = 'X_EndP'
HW_ENDP_Y = 'Y_EndP'
# Subcatchments:
SC_ID = 'sc_ID'
# Debris flow constituents lookup:
PCLE_CTUENT_NAME = 'Particulate constituent'
AVG_CTUENT_MGPKG = 'Average amount (mgkg-1)'
# HF lookup:
HF_ARID_IDX_THRESH = 'AI'
HF_DNBR_THRESH = 'dNBR'
HF_YEARS_THRESH = 'years'
HF_GRADIENT_THRESH = 'slope'
HF_I12_CRIT = 'I12_crit_mean'
# Summary stats (fire_impact_data) table:
adjusted_suffix = '_adjusted'
year_suffix = '_Year_'
ARID_MEAN = 'Aridity_mean'
ARID_MEAN_ADJ = ARID_MEAN + adjusted_suffix
DNBR_MEAN = 'dNBR_mean'
DNBR_MEAN_ADJ = DNBR_MEAN + adjusted_suffix
SLOPE_DEG_MEAN = 'Slope_mean'
SLOPE_DEG_MEAN_ADJ = SLOPE_DEG_MEAN + adjusted_suffix
I12_CRIT_Y = HF_I12_CRIT + year_suffix

# ------- Output file names: ------------------------------------------
# RUSLE erosion:
RUSLE_OP_PEAK_NAME = 'peak_erosion'
RUSLE_OP_TOTAL_NAME = 'erosion_total'
RUSLE_OP_TIMESERIES_NAME = 'erosion_daily_time_series'

# Sediment delivered to streams:
DELIVERED_OP_PEAK_NAME = 'peak_delivered'
DELIVERED_OP_TOTAL_NAME = 'delivered_total'

RUSLE_OUTPUT_RASTER_NAMES = [
    RUSLE_OP_PEAK_NAME,
    RUSLE_OP_TOTAL_NAME,
    DELIVERED_OP_PEAK_NAME,
    DELIVERED_OP_TOTAL_NAME,
]
RUSLE_OP_TIMESERIES_NAME = 'erosion_daily_time_series'
# Debris flow:
DEBRIS_OP_TIMESERIES_NAME = 'debris_daily_time_series'
# Subcatchment summary outputs (aggregated from headwater/raster
# results):
RUSLE_SC_SUMMARY_NAME = 'rusle_subcatchment_summary'
DEBRIS_SC_SUMMARY_NAME = 'DebrisFlowData_subcatchments'
# Rainfall:
RAIN_DAILY_DEPTH_TIMESERIES_NAME = 'rain_depth_daily_time_series'
# Persisted stochastic rainfall for an ensemble, written under
# Ensembles/<ensemble>/. get_rainfall_replicates caches its output here and
# reuses it on repeat runs; save_ensemble_run / load_ensemble_rainfall use
# the same file.
RAINFALL_NAME = 'rainfall.nc'

# ------- Directory folder names: -------------------------------------
TOPOGRAPHY_FOLDER_NAME = 'Topography'
FIRE_SEVERITY_FOLDER_NAME = 'FireSeverity'
SOILS_FOLDER_NAME = 'Soils'
ERODIBILITY_FOLDER_NAME = 'Erodibility'
DELIVERY_FOLDER_NAME = 'Delivery'
SUBCATCHMENTS_FOLDER_NAME = 'Subcatchments'
RESULTS_FOLDER_NAME = 'Results'
RESULTS_BASELINE_FOLDER_NAME = 'Results_baseline'
# Standard subfolders created inside every catchment directory. These hold
# fire-independent, catchment-scope data. FireSeverity is not included here:
# it is per-event and created under Events/<event>/ by calculate_fire_severity.
# Results (and Results_baseline, DebrisFlow) are per-run and created under
# Runs/<event>/<ensemble>/ by the simulation, so they are not pre-created at
# catchment scope either.
PER_CATCHMENT_FOLDERS = [
    TOPOGRAPHY_FOLDER_NAME,
    SOILS_FOLDER_NAME,
    ERODIBILITY_FOLDER_NAME,
    DELIVERY_FOLDER_NAME,
    SUBCATCHMENTS_FOLDER_NAME,
    ]
