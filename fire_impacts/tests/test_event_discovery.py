"""
Discovering what exists, and saying so when it does not.

A RunContext can be built for an event that does not exist yet — a user
may reasonably create the context before running the prep that creates
the event. The cost is that the first failure comes from whichever
per-event file happens to be read first, and its message describes the
wrong problem.
"""

import pytest

from fire_impacts.context import RunContext
from fire_impacts.pre.project import FireImpactsProject


@pytest.fixture
def project(tmp_path):
    proj = FireImpactsProject(str(tmp_path / 'proj'), exist_ok=False)
    proj.catchments.append('Big-River')
    proj._write()
    return proj


def _make(project, catchment, folder, *names):
    from pathlib import Path
    for name in names:
        (Path(project.catchment_path(catchment)) / folder / name).mkdir(
            parents=True, exist_ok=True)


class TestDiscovery:
    """The counterpart of proj.catchments for the two on-disk axes."""

    def test_no_events_yet(self, project):
        assert project.events('Big-River') == []

    def test_events_are_listed_sorted(self, project):
        _make(project, 'Big-River', 'Events', '2020_fire', '2019_fire')
        assert project.events('Big-River') == ['2019_fire', '2020_fire']

    def test_ensembles_are_listed(self, project):
        _make(project, 'Big-River', 'Ensembles', 'historical', 'rcp85')
        assert project.ensembles('Big-River') == ['historical', 'rcp85']

    def test_the_catchment_may_be_omitted_when_there_is_only_one(
            self, project):
        _make(project, 'Big-River', 'Events', '2019_fire')
        assert project.events() == ['2019_fire']

    def test_omitting_it_with_several_catchments_raises(self, project):
        project.catchments.append('Second')
        with pytest.raises(ValueError, match='name the one to query'):
            project.events()

    def test_files_are_not_mistaken_for_events(self, project):
        from pathlib import Path
        events = Path(project.catchment_path('Big-River')) / 'Events'
        events.mkdir(parents=True)
        (events / 'notes.txt').write_text('x')
        assert project.events('Big-River') == []

    def test_list_events_delegates(self, project):
        from fire_impacts.sim import list_events
        _make(project, 'Big-River', 'Events', '2019_fire')
        assert list_events(project, 'Big-River') == project.events('Big-River')

    def test_list_ensembles_delegates(self, project):
        from fire_impacts.sim import list_ensembles
        _make(project, 'Big-River', 'Ensembles', 'historical')
        assert list_ensembles(project, 'Big-River') == \
            project.ensembles('Big-River')


class TestMissingEventIsReportedAsSuch:
    """A context for a non-existent event used to fail with 'Run
    calculate_fire_severity for event X first', which reads as though the
    event exists and one step is outstanding."""

    def test_a_context_can_still_be_built_for_a_future_event(self, project):
        # Deliberate: the prep that creates the event may run after the
        # context does. So the check cannot live in construction.
        ctx = RunContext.solo_run(
            project, event='not_yet', ensemble='historical')
        assert ctx.event == 'not_yet'

    def test_simulation_period_names_the_missing_event(self, project):
        ctx = RunContext.solo_run(
            project, event='typo_fire', ensemble='historical')
        with pytest.raises(FileNotFoundError, match="Event 'typo_fire' does not exist"):
            ctx.simulation_period()

    def test_the_error_lists_the_events_that_do_exist(self, project):
        """What actually resolves a typo."""
        _make(project, 'Big-River', 'Events', '2019_fire', '2020_fire')
        ctx = RunContext.solo_run(
            project, event='2019_fyre', ensemble='historical')
        with pytest.raises(FileNotFoundError) as excinfo:
            ctx.simulation_period()
        assert "'2019_fire'" in str(excinfo.value)
        assert "'2020_fire'" in str(excinfo.value)

    def test_with_no_events_it_points_at_the_prep_notebook(self, project):
        ctx = RunContext.solo_run(
            project, event='anything', ensemble='historical')
        with pytest.raises(FileNotFoundError, match='PrepareData'):
            ctx.simulation_period()

    def test_the_event_definition_accessor_reports_it_too(self, project):
        ctx = RunContext.solo_run(
            project, event='typo_fire', ensemble='historical')
        with pytest.raises(FileNotFoundError, match='does not exist'):
            ctx.event_definition()

    def test_no_misleading_breakpoint_warning_is_logged_first(
            self, project, caplog):
        """event_definition warned about re-running compute_adjusted_k_c
        before failing — a second wrong signal for a missing event."""
        ctx = RunContext.solo_run(
            project, event='typo_fire', ensemble='historical')
        with caplog.at_level('WARNING'):
            with pytest.raises(FileNotFoundError):
                ctx.event_definition()
        assert 'compute_adjusted_k_c' not in caplog.text

    def test_an_existing_event_still_reports_the_real_next_step(
            self, project):
        """The original message is right when the event does exist."""
        _make(project, 'Big-River', 'Events', '2019_fire')
        ctx = RunContext.solo_run(
            project, event='2019_fire', ensemble='historical')
        with pytest.raises(FileNotFoundError,
                           match='Run calculate_fire_severity'):
            ctx.simulation_period()
