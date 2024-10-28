'''
This python module encompasses modelling routines for assessing the impacts of bushfires on the
water quality characteristics of catchments.

The module is tested in Australian catchments and is organised around data that is broadly available across Australia.

The module is divided into the following sub-modules:

* fire_impacts.pre: Pre-processing routines for extracting topographic, soil, and fire severity data.

The module relies on a standard directory structure, managed and implemented by the FireImpactsProject class.
'''

from .pre import *