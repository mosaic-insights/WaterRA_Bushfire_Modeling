"""
Telling the user when a run is paying full price.

A recorder that forces a full grid every timestep costs roughly a hundred
times a deferred one. That is legitimate for some recorders, but it should
never be a surprise.
"""

import numpy as np
import pandas as pd

from fire_impacts.sim.rusle import _warn_about_materialising
from fire_impacts.sim.scaled_grid import MaterialisationCounter, ScaledGrid


def greedy(timestep, **data):
    """A recorder that insists on a real grid."""
    return float(np.nansum(data['RUSLE']))


def frugal(timestep, **data):
    """A recorder that only looks at the scale."""
    return data['RUSLE'].scale[0]


def deferred(counter):
    return ScaledGrid(np.array([2.0]), np.ones((2, 3)), counter=counter)


class TestNamingTheCulprit:

    def test_names_the_recorder_that_materialises(self, caplog):
        counter = MaterialisationCounter()
        with caplog.at_level('WARNING'):
            _warn_about_materialising(
                {'greedy': greedy, 'frugal': frugal},
                pd.Timestamp('2020-01-01'),
                {'RUSLE': deferred(counter)},
                counter,
            )

        assert 'greedy' in caplog.text
        assert 'frugal' not in caplog.text

    def test_says_nothing_when_every_recorder_defers(self, caplog):
        counter = MaterialisationCounter()
        with caplog.at_level('WARNING'):
            _warn_about_materialising(
                {'frugal': frugal},
                pd.Timestamp('2020-01-01'),
                {'RUSLE': deferred(counter)},
                counter,
            )

        assert caplog.text == ''


class TestCounterIsolation:

    def test_counter_instances_tally_independently(self):
        # Two MaterialisationCounter objects do not share state. This is
        # not the per-run guarantee (a shared instance would still fail
        # this if handed to both) -- that is covered by
        # test_each_run_gets_its_own_materialisation_counter in
        # test_integration_pipeline.py, which checks that
        # run_usle_simulation itself mints a fresh counter per call.
        a, b = MaterialisationCounter(), MaterialisationCounter()
        np.asarray(deferred(a))

        assert (a.count, b.count) == (1, 0)
