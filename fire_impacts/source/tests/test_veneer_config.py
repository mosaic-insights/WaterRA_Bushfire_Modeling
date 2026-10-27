"""
Unit tests for veneer_config module, focusing on _detect_parameter function.
"""

import ast
import sys
import pytest
import logging
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from fire_impacts.source.veneer_config import (
    _detect_parameter, create_file_data_source, create_veneer_data_sources,
    ensure_constituent, restricted_data_source_name, run_model_simulation,
    run_succeeded,
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


class TestDetectParameter:
    """Test suite for _detect_parameter function."""
    
    def test_substring_match_exact(self):
        """Test exact substring match."""
        getter_func = MagicMock(return_value=['TSS', 'Nitrogen', 'Phosphorus'])
        candidates = ['TSS', 'Sediment']
        
        result = _detect_parameter(getter_func, candidates, 'constituents')
        assert result == 'TSS'
    
    def test_substring_match_falls_through_to_later_candidate(self):
        """A candidate that matches nothing gives way to the next one."""
        getter_func = MagicMock(return_value=['Total Suspended Solids', 'Nitrogen', 'Phosphorus'])
        # 'TSS' is not a substring of 'Total Suspended Solids', so the
        # scan should move on to 'Solids'.
        candidates = ['TSS', 'Solids']

        result = _detect_parameter(getter_func, candidates, 'constituents')
        assert result == 'Total Suspended Solids'

    def test_substring_match_case_insensitive(self):
        """Test case-insensitive substring matching."""
        getter_func = MagicMock(return_value=['TSS_Load', 'Nitrogen', 'Phosphorus'])
        candidates = ['tss']  # lowercase candidate
        
        result = _detect_parameter(getter_func, candidates, 'constituents')
        assert result == 'TSS_Load'
    
    def test_substring_match_partial_in_option(self):
        """Test candidate substring appears within option name."""
        getter_func = MagicMock(return_value=['Forested_Urban', 'Urban', 'Water'])
        candidates = ['Forested']
        
        result = _detect_parameter(getter_func, candidates, 'functional units')
        assert result == 'Forested_Urban'
    
    def test_substring_match_priority_order(self):
        """Test that first matching candidate takes priority."""
        getter_func = MagicMock(return_value=['TSS_Sediment', 'Sediment', 'Contaminant'])
        candidates = ['Sediment', 'TSS', 'Contaminant']
        
        result = _detect_parameter(getter_func, candidates, 'constituents')
        # 'Sediment' candidate should match first (appears in 'TSS_Sediment')
        assert result == 'TSS_Sediment'
    
    def test_substring_match_first_available_option(self):
        """Test returns first matching available option when multiple candidates match."""
        getter_func = MagicMock(return_value=['TSS_Load', 'TSS_Concentration', 'Nitrogen'])
        candidates = ['TSS']
        
        result = _detect_parameter(getter_func, candidates, 'constituents')
        assert result == 'TSS_Load'  # First option containing 'TSS'
    
    def test_no_match_fallback_to_first(self):
        """Test fallback to first available option when no candidate matches."""
        getter_func = MagicMock(return_value=['Nitrogen', 'Phosphorus', 'Potassium'])
        candidates = ['TSS', 'Sediment', 'Contaminant']
        
        result = _detect_parameter(getter_func, candidates, 'constituents')
        assert result == 'Nitrogen'
    
    def test_filter_function_applied(self):
        """Test that filter function is applied to available options."""
        getter_func = MagicMock(return_value=['Forested', 'Urban', 'Water'])
        candidates = ['Forested']
        
        def filter_out_water(options):
            return [opt for opt in options if opt != 'Water']
        
        result = _detect_parameter(
            getter_func, candidates, 'functional units', filter_func=filter_out_water
        )
        assert result == 'Forested'
    
    def test_filter_function_with_no_match(self):
        """Test fallback when filter removes all but one option."""
        getter_func = MagicMock(return_value=['Forested', 'Urban', 'Water'])
        candidates = ['Nitrogen', 'Phosphorus']
        
        def filter_out_water(options):
            return [opt for opt in options if opt != 'Water']
        
        result = _detect_parameter(
            getter_func, candidates, 'functional units', filter_func=filter_out_water
        )
        # Should return first available after filtering
        assert result == 'Forested'
    
    def test_error_when_no_options_available(self):
        """Test error is raised when getter returns empty list."""
        getter_func = MagicMock(return_value=[])
        candidates = ['TSS', 'Sediment']
        
        with pytest.raises(ValueError, match="No constituents found in model"):
            _detect_parameter(getter_func, candidates, 'constituents')
    
    def test_error_when_filter_removes_all_options(self):
        """Test error is raised when filter removes all options."""
        getter_func = MagicMock(return_value=['Water', 'Water', 'Water'])
        candidates = ['Forested', 'Urban']
        
        def filter_out_water(options):
            return [opt for opt in options if opt != 'Water']
        
        with pytest.raises(ValueError, match="No functional units found in model"):
            _detect_parameter(
                getter_func, candidates, 'functional units', filter_func=filter_out_water
            )
    
    def test_error_when_getter_raises_exception(self):
        """Test error propagation when getter function fails."""
        def failing_getter():
            raise RuntimeError("Connection failed")
        
        candidates = ['TSS']
        
        with pytest.raises(RuntimeError, match="Connection failed"):
            _detect_parameter(failing_getter, candidates, 'constituents')
    
    def test_real_world_example_tss_variants(self):
        """Real-world example: matching TSS across different naming conventions."""
        getter_func = MagicMock(return_value=[
            'TSS_Daily',
            'Total_Suspended_Sediment',
            'Nitrogen',
            'Phosphorus'
        ])
        candidates = ['TSS', 'Sediment', 'Contaminant']
        
        result = _detect_parameter(getter_func, candidates, 'constituents')
        # Should match 'TSS_Daily' with 'TSS' candidate
        assert result == 'TSS_Daily'
    
    def test_real_world_example_forested_variants(self):
        """Real-world example: matching Forested across different naming conventions."""
        getter_func = MagicMock(return_value=[
            'Forested_Native',
            'Urban_Built',
            'Agricultural',
            'Water'
        ])
        candidates = ['Forested', 'Forest', 'Urban']
        
        def filter_water(options):
            return [opt for opt in options if opt != 'Water']
        
        result = _detect_parameter(
            getter_func, candidates, 'functional units', filter_func=filter_water
        )
        assert result == 'Forested_Native'
    
    def test_candidate_with_special_characters(self):
        """Test candidates and options with special characters."""
        getter_func = MagicMock(return_value=['TSS (kg/h)', 'Sediment (mg/L)', 'Nitrogen'])
        candidates = ['TSS', 'Sediment']
        
        result = _detect_parameter(getter_func, candidates, 'constituents')
        assert result == 'TSS (kg/h)'
    
    def test_empty_candidate_list(self):
        """Test with empty candidate list - should fallback to first available."""
        getter_func = MagicMock(return_value=['TSS', 'Nitrogen', 'Phosphorus'])
        candidates = []
        
        result = _detect_parameter(getter_func, candidates, 'constituents')
        assert result == 'TSS'
    
    def test_single_option_single_candidate(self):
        """Test with single option and matching candidate."""
        getter_func = MagicMock(return_value=['TSS'])
        candidates = ['TSS']
        
        result = _detect_parameter(getter_func, candidates, 'constituents')
        assert result == 'TSS'
    
    def test_case_sensitivity_mixed(self):
        """Test case-insensitive matching with mixed case in both candidate and option."""
        getter_func = MagicMock(return_value=['TsS_Load', 'Nitrogen', 'PHOSPHORUS'])
        candidates = ['tss']
        
        result = _detect_parameter(getter_func, candidates, 'constituents')
        assert result == 'TsS_Load'
    
    def test_unicode_characters(self):
        """Test with unicode characters in options and candidates."""
        getter_func = MagicMock(return_value=['SÉ_Suspended', 'Nitrogène', 'Phosphore'])
        candidates = ['sé', 'nitrogène']
        
        result = _detect_parameter(getter_func, candidates, 'constituents')
        assert result == 'SÉ_Suspended'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def _veneer_reporting(status):
    """A Veneer stand-in whose next run reports the given status."""
    v = MagicMock()
    v.retrieve_run.return_value = {'Status': status, 'RunLog': ['a line']}
    return v


class TestRunStatus:
    """Source reports a run's outcome as the name of its RunResultType:
    'RunSuccess' when it ran, something else (e.g. 'RunIncomplete') when
    it did not."""

    def test_a_successful_run_is_recognised(self):
        assert run_succeeded({'Status': 'RunSuccess'})

    @pytest.mark.parametrize('status', ['RunIncomplete', 'Finished', ''])
    def test_anything_else_is_not_a_success(self, status):
        assert not run_succeeded({'Status': status})

    def test_a_result_with_no_status_is_not_a_success(self):
        assert not run_succeeded({})

    def test_a_successful_run_is_not_reported_as_a_failure(self, caplog):
        run_model_simulation(_veneer_reporting('RunSuccess'))
        assert 'did not complete' not in caplog.text

    def test_an_unsuccessful_run_is_reported_with_its_status(self, caplog):
        run_model_simulation(_veneer_reporting('RunIncomplete'))
        assert 'did not complete' in caplog.text
        assert 'RunIncomplete' in caplog.text


class TestInputSet:
    """Source names its standard input set 'Default Input Set', and a
    project may have others. Guessing a name selects nothing or the wrong
    one, so none is selected unless the caller names it."""

    def test_no_input_set_leaves_source_on_the_one_already_selected(self):
        v = _veneer_reporting('RunSuccess')
        run_model_simulation(v)
        v.model.simulation.select_input_set.assert_not_called()

    def test_a_named_input_set_is_selected_before_the_run(self):
        v = _veneer_reporting('RunSuccess')
        run_model_simulation(v, input_set='Default Input Set')

        v.model.simulation.select_input_set.assert_called_once_with(
            'Default Input Set')
        names = [name for name, _, _ in v.mock_calls]
        assert (names.index('model.simulation.select_input_set')
                < names.index('run_model'))


def _gui_name(name):
    """What the Source interface makes of a new data source's name: its
    data sources tree renames each group it shows with DataManager's
    RestrictName rule, and the rename sticks."""
    for c in '.@>,\\':
        name = name.replace(c, '_')
    return name[:200].strip(' ')


class FakeGroup:
    def __init__(self, name):
        self.Name = name


class FakeDataManager:
    def __init__(self, names=()):
        self.DataGroups = [FakeGroup(n) for n in names]

    def RemoveGroup(self, group):
        self.DataGroups.remove(group)


class FakeSource:
    """Stands in for Veneer and the Source behind it.

    Data sources are named as Veneer and Source name them: after the path
    for a file, put through the interface's renaming rule when Source has
    its interface open (gui=True). Scripts sent to Source are run against a
    fake data manager, and units are looked up by name, the way veneer-py
    looks them up."""

    def __init__(self, gui=True, groups=(), rename=_gui_name):
        self.gui = gui
        self.rename = rename
        self.dm = FakeDataManager(groups)
        self.created = []
        self.scripts = []
        self.units = {}
        self.model = MagicMock()
        self.model.run_script.side_effect = self._run_script
        self.model.set_data_source_units.side_effect = self._set_units

    @property
    def names(self):
        return [g.Name for g in self.dm.DataGroups]

    def create_data_source(self, name, data=None, units=None,
                           reload_on_run=False, **kwargs):
        self.created.append(dict(name=name, data=data, units=units,
                                 reload_on_run=reload_on_run))
        # Veneer replaces a data source of exactly that name, and the
        # replacement keeps the name as given.
        existing = [g for g in self.dm.DataGroups if g.Name == name]
        for g in existing:
            self.dm.RemoveGroup(g)
        shown_as = self.rename(name) if self.gui and not existing else name
        self.dm.DataGroups.append(FakeGroup(shown_as))
        if data is None and units is not None:
            self._set_units(name, units)

    def _set_units(self, name, units):
        if name not in self.names:
            raise AssertionError('Unknown data source: ' + name)
        self.units[name] = units

    def _run_script(self, script, init=False):
        self.scripts.append(script)
        scope = {'scenario': SimpleNamespace(
            Network=SimpleNamespace(DataManager=self.dm))}
        try:
            exec(script, scope)
        except Exception as e:
            return {'Exception': str(e), 'Response': None}
        return {'Exception': None,
                'Response': {'Value': scope.get('result')}}


class FakeSourceThatLoadsNothing(FakeSource):
    def create_data_source(self, name, data=None, units=None, **kwargs):
        self.created.append(dict(name=name, data=data, units=units))


def _string_constants(script):
    """Every string literal in a script, as IronPython would read it."""
    return {node.value for node in ast.walk(ast.parse(script))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)}


class TestRestrictedDataSourceName:
    """The Source interface renames data sources, replacing characters it
    reserves for its own paths with underscores."""

    def test_reserved_characters_become_underscores(self):
        assert (restricted_data_source_name(r'C:\Runs\a.b@c>d,e.csv')
                == 'C:_Runs_a_b_c_d_e_csv')

    def test_other_characters_are_kept(self):
        name = 'C:/runs/fire-2001 (rep #1):tss'
        assert restricted_data_source_name(name) == name

    def test_long_names_are_cut_to_200_characters(self):
        assert restricted_data_source_name('x' * 250) == 'x' * 200

    def test_surrounding_spaces_are_trimmed_after_cutting(self):
        name = ' ' + 'x' * 198 + '  y'
        assert restricted_data_source_name(name) == 'x' * 198


class TestCreateFileDataSource:
    """Veneer names a file-based data source after the path it was given,
    and the Source interface then renames it, replacing every backslash
    and full stop with an underscore. Neither is the name the notebook
    goes on to assign from, and assigning from a name Source does not have
    fails with a NullReferenceException. create_file_data_source gives the
    data source the name the caller asked for."""

    PATH = r'C:\Projects\Runs\2001_fire\stochastic\source_inputs\fire_tss.csv'

    @pytest.mark.parametrize('gui', [True, False])
    def test_the_data_source_ends_up_with_the_name_asked_for(self, gui):
        v = FakeSource(gui=gui)
        create_file_data_source(v, 'fire_tss', self.PATH)
        assert v.names == ['fire_tss']

    def test_source_is_asked_to_load_the_file_and_reload_it(self):
        v = FakeSource()
        create_file_data_source(v, 'fire_tss', self.PATH)

        [call] = v.created
        assert call['name'] == self.PATH
        assert call['data'] is None
        assert call['reload_on_run'] is True

    @pytest.mark.parametrize('gui', [True, False])
    def test_units_are_applied_under_the_new_name(self, gui):
        """veneer-py applies a file's units by looking the data source up
        under the path - which the interface has already renamed."""
        v = FakeSource(gui=gui)
        create_file_data_source(v, 'fire_tss', self.PATH, units='kg/day')
        assert v.units == {'fire_tss': 'kg/day'}

    def test_no_units_are_applied_unless_asked_for(self):
        v = FakeSource()
        create_file_data_source(v, 'fire_tss', self.PATH)
        assert v.units == {}

    def test_the_interface_rule_comes_from_source_when_it_has_one(
            self, monkeypatch):
        """A Source version whose rule differs from the one built in here
        still has its data source found."""
        def other_rule(name):
            return name.replace('\\', '_').replace('_', '-')

        dm_module = ModuleType('RiverSystem.DataManagement.DataManager')
        dm_module.DataManager = SimpleNamespace(RestrictName=other_rule)
        clr = ModuleType('clr')
        clr.AddReference = lambda name: None
        for name, module in [
                ('clr', clr),
                ('RiverSystem', ModuleType('RiverSystem')),
                ('RiverSystem.DataManagement',
                 ModuleType('RiverSystem.DataManagement')),
                ('RiverSystem.DataManagement.DataManager', dm_module)]:
            monkeypatch.setitem(sys.modules, name, module)

        v = FakeSource(rename=other_rule)
        create_file_data_source(v, 'fire_tss', self.PATH, units='kg/day')
        assert v.names == ['fire_tss']

    def test_any_other_data_source_with_that_name_is_replaced(self):
        """Part B leaves an in-memory data source of the same name, and a
        failed assignment leaves an empty one."""
        v = FakeSource(groups=['fire_tss', 'stochastic_rain'])
        create_file_data_source(v, 'fire_tss', self.PATH)
        assert sorted(v.names) == ['fire_tss', 'stochastic_rain']

    def test_one_left_over_from_an_earlier_attempt_is_removed(self):
        """A run that failed after creating the data source leaves it
        behind under the interface's version of the path."""
        v = FakeSource(groups=[_gui_name(self.PATH), 'stochastic_rain'])
        create_file_data_source(v, 'fire_tss', self.PATH)
        assert sorted(v.names) == ['fire_tss', 'stochastic_rain']

    def test_a_non_ascii_path_reaches_source_as_written(self):
        """IronPython 2 leaves a backslash-u escape in an ordinary string
        as literal characters, so an accented folder name has to arrive as
        itself."""
        path = 'C:\\Users\\Jos\u00e9\\fire_tss.csv'
        v = FakeSource()
        create_file_data_source(v, 'fire_tss', path)

        [script] = v.scripts
        assert '\\u00e9' not in script
        assert path in _string_constants(script)
        assert v.names == ['fire_tss']

    def test_a_file_source_could_not_load_is_reported(self):
        v = FakeSourceThatLoadsNothing()
        with pytest.raises(Exception, match='No data source was created'):
            create_file_data_source(v, 'fire_tss', self.PATH)

    @pytest.mark.parametrize('name', ['fire.tss', 'fire,tss', 'runs\\tss'])
    def test_a_name_the_interface_would_change_is_refused(self, name):
        v = FakeSource()
        with pytest.raises(ValueError) as raised:
            create_file_data_source(v, name, self.PATH)
        assert name in str(raised.value)
        assert v.created == []

    def test_it_returns_the_name_to_assign_from(self):
        v = FakeSource()
        assert create_file_data_source(v, 'fire_tss', self.PATH) == 'fire_tss'


class TestCreateVeneerDataSources:
    def test_a_name_the_interface_would_change_is_refused(self):
        v = MagicMock()
        with pytest.raises(ValueError, match='fire.tss'):
            create_veneer_data_sources(
                v, pd.DataFrame(), pd.DataFrame(), tss_source_name='fire.tss')
        v.create_data_source.assert_not_called()


class FakeConstituents:
    """The constituent list of a Source model, and veneer-py's calls on it."""

    def __init__(self, constituents=(), adds=True):
        self.constituents = list(constituents)
        self.adds = adds
        self.added = []
        self.model = MagicMock()
        self.model.get_constituents.side_effect = (
            lambda: list(self.constituents))
        self.model.add_constituent.side_effect = self._add

    def _add(self, name):
        self.added.append(name)
        if self.adds:
            self.constituents.append(name)
        return {'Exception': None, 'Response': None}


class TestEnsureConstituent:
    def test_a_named_constituent_the_model_has_is_used(self):
        v = FakeConstituents(['Salt', 'TSS'])
        assert ensure_constituent(v, 'TSS') == 'TSS'
        assert v.added == []

    def test_a_named_constituent_the_model_lacks_is_created(self, caplog):
        v = FakeConstituents(['Salt'])
        with caplog.at_level(logging.INFO):
            assert ensure_constituent(v, 'TSS') == 'TSS'
        assert v.added == ['TSS']
        assert "Created constituent 'TSS'" in caplog.text

    def test_a_named_constituent_is_matched_regardless_of_case(self, caplog):
        """Rather than create a second constituent that differs only in
        case."""
        v = FakeConstituents(['Salt', 'TSS'])
        assert ensure_constituent(v, 'tss') == 'TSS'
        assert v.added == []
        assert "'TSS'" in caplog.text

    def test_left_unnamed_a_likely_constituent_is_picked(self):
        v = FakeConstituents(['Salt', 'Fine Sediment'])
        assert ensure_constituent(v) == 'Fine Sediment'
        assert v.added == []

    def test_left_unnamed_in_a_model_without_constituents_tss_is_created(
            self, caplog):
        v = FakeConstituents([])
        with caplog.at_level(logging.INFO):
            assert ensure_constituent(v) == 'TSS'
        assert v.added == ['TSS']
        assert "Created constituent 'TSS'" in caplog.text

    def test_left_unnamed_an_unrelated_constituent_is_not_taken_over(
            self, caplog):
        """Configuring the Load Distributor replaces a constituent's
        generation models, so a salt model is not the place for sediment."""
        v = FakeConstituents(['Salt', 'Nitrogen'])
        assert ensure_constituent(v) == 'TSS'
        assert v.added == ['TSS']
        assert 'Salt' in caplog.text and 'source.constituent' in caplog.text

    def test_a_constituent_source_did_not_create_is_reported(self):
        v = FakeConstituents([], adds=False)
        with pytest.raises(RuntimeError, match='TSS'):
            ensure_constituent(v, 'TSS')
