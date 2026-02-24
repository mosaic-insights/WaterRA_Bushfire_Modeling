'''
This module contains the classes and functions that are used to prepare the data for the fire_impacts module.

The module includes the following workflows, implemented as sub-modules:

* fire_impacts.pre.topography: Perform catchment (headwater) delineation from a DEM.
* fire_impacts.pre.soil: Download soil data from the Soil and Landscape Grid of Australia.
* fire_impacts.pre.severity: Calculate fire severity indices (NBR, dNBR) before and after a fire.
* fire_impacts.pre.mask_dnbr: Mask dNBR to specific DEA land cover classes (e.g., Natural Terrestrial Vegetation).
* fire_impacts.pre.project: Objects representing the project folder structure for a fire impacts study.
* fire_impacts.pre.utils: Utility functions for the fire impacts pre-processing module.
'''

from .project import FireImpactsProject, find_all_shapefiles
