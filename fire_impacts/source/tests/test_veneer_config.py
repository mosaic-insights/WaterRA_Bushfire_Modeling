"""
Unit tests for veneer_config module, focusing on _detect_parameter function.
"""

import pytest
import logging
from pathlib import PureWindowsPath
from unittest.mock import MagicMock

import pandas as pd

from fire_impacts.source.veneer_config import (
    _detect_parameter, create_file_data_source, create_veneer_data_sources,
    ensure_constituent, restricted_data_source_name, run_model_simulation,
    run_succeeded, save_model, write_replicate_csvs,
)
from fire_impacts.study import SOURCE_TIMESTEPS

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


class FakeVeneer:
    """Stands in for veneer-py (0.3.0 or later) and the Source behind it.

    create_data_source names a file data source itself: data_source_name,
    or else the file's stem, put through the interface's renaming rule. It
    replaces any data source already of that name, and returns the name it
    kept. Units are looked up by name, the way veneer-py looks them up.

    keeps: the name veneer-py reports it kept, when that is not the one
    asked for - an older Veneer with "Allow Scripts" off keeps the one
    Source gave the data source."""

    def __init__(self, groups=(), keeps=None):
        self.names = list(groups)
        self.keeps = keeps
        self.created = []
        self.units = {}
        self.model = MagicMock()
        self.model.set_data_source_units.side_effect = self._set_units

    def create_data_source(self, name, data=None, units=None, precision=3,
                           reload_on_run=False, data_source_name=None):
        self.created.append(dict(name=name, data=data, units=units,
                                 reload_on_run=reload_on_run,
                                 data_source_name=data_source_name))
        if data_source_name is None:
            data_source_name = (PureWindowsPath(name).stem
                                if data is None else name)
        kept = self.keeps or restricted_data_source_name(data_source_name)
        self.names = [n for n in self.names if n != kept] + [kept]
        if units is not None:
            self._set_units(kept, units)
        return kept

    def _set_units(self, name, units):
        if name not in self.names:
            raise AssertionError('Unknown data source: ' + name)
        self.units[name] = units


class FakeOldVeneerPy(FakeVeneer):
    """veneer-py 0.2.1 and earlier, whose create_data_source takes no
    data_source_name."""

    def create_data_source(self, name, data=None, units=None, precision=3,
                           reload_on_run=False):
        raise AssertionError('not reached: the call itself is refused')


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
    """veneer-py names a file data source after the file's stem unless told
    otherwise, and the notebook goes on to assign from the name it asked
    for: assigning from a name Source does not have fails with a
    NullReferenceException. create_file_data_source has veneer-py use that
    name, and checks it did."""

    PATH = r'C:\Projects\Runs\2001_fire\stochastic\source_inputs\fire_tss.csv'
    OTHER_STEM = r'C:\Projects\Runs\2001_fire\source_inputs\replicate.csv'

    def test_the_data_source_ends_up_with_the_name_asked_for(self):
        v = FakeVeneer()
        create_file_data_source(v, 'fire_tss', self.OTHER_STEM)
        assert v.names == ['fire_tss']

    def test_veneer_py_is_asked_to_load_the_file_and_reload_it(self):
        v = FakeVeneer()
        create_file_data_source(v, 'fire_tss', self.OTHER_STEM)

        [call] = v.created
        assert call['name'] == self.OTHER_STEM
        assert call['data'] is None
        assert call['data_source_name'] == 'fire_tss'
        assert call['reload_on_run'] is True

    def test_no_script_is_sent(self):
        """With a Veneer of protocol 20261027 or later, nothing here needs
        Veneer's "Allow Scripts"."""
        v = FakeVeneer()
        create_file_data_source(v, 'fire_tss', self.PATH)
        v.model.run_script.assert_not_called()

    def test_any_other_data_source_with_that_name_is_replaced(self):
        """Part B leaves an in-memory data source of the same name, and a
        failed assignment leaves an empty one."""
        v = FakeVeneer(groups=['fire_tss', 'stochastic_rain'])
        create_file_data_source(v, 'fire_tss', self.PATH)
        assert sorted(v.names) == ['fire_tss', 'stochastic_rain']

    def test_units_are_applied_under_the_name(self):
        v = FakeVeneer()
        create_file_data_source(
            v, 'fire_tss', self.OTHER_STEM, units='kg/day',
            reload_on_run=False)
        assert v.units == {'fire_tss': 'kg/day'}

    def test_no_units_are_applied_unless_asked_for(self):
        v = FakeVeneer()
        create_file_data_source(v, 'fire_tss', self.PATH)
        assert v.units == {}

    def test_units_for_a_file_source_reloads_are_refused(self):
        """Source re-reads each column's units from the CSV header on every
        reload. A bare column name that parses as a unit - 'A', amperes -
        then clashes with units applied after creation, and every run fails:
        'Loaded units are not commensurate with the previously defined
        units'. Units in the headers work whatever the columns are called,
        and subcatchment names are not ours to choose."""
        v = FakeVeneer()
        with pytest.raises(ValueError) as raised:
            create_file_data_source(v, 'fire_tss', self.PATH, units='kg/day')
        message = str(raised.value)
        assert "'<column> (kg/day)'" in message
        assert 'parses as a unit' in message
        assert v.created == []

    @pytest.mark.parametrize('name', ['fire.tss', 'fire,tss', 'runs\\tss'])
    def test_a_name_the_interface_would_change_is_refused(self, name):
        v = FakeVeneer()
        with pytest.raises(ValueError) as raised:
            create_file_data_source(v, name, self.PATH)
        assert name in str(raised.value)
        assert v.created == []

    def test_a_data_source_left_under_another_name_is_reported(self):
        """An older Veneer with "Allow Scripts" off leaves the data source
        with the name Source gave it."""
        kept = restricted_data_source_name(self.PATH)
        v = FakeVeneer(keeps=kept)
        with pytest.raises(Exception) as raised:
            create_file_data_source(v, 'fire_tss', self.PATH)
        message = str(raised.value)
        assert repr(kept) in message
        assert "'fire_tss'" in message
        assert 'Allow Scripts' in message

    def test_an_older_veneer_py_is_reported(self):
        v = FakeOldVeneerPy()
        with pytest.raises(TypeError, match='veneer-py 0.3.0 or later'):
            create_file_data_source(v, 'fire_tss', self.PATH)

    def test_it_returns_the_name_to_assign_from(self):
        v = FakeVeneer()
        assert create_file_data_source(v, 'fire_tss', self.PATH) == 'fire_tss'


class TestWriteReplicateCsvs:
    """Part A's CSVs carry their units in their column headers - "A (ML/d)"
    loads as column "A" in ML/d. Source re-reads them on every reload, so
    they hold whatever the subcatchments are called; units applied after
    creation break the reload of any column whose bare name parses as a
    unit."""

    @staticmethod
    def _replicate(columns=('SC1', 'SC2')):
        index = pd.date_range('2001-01-01', periods=3, freq='D', name='Date')
        loads = pd.DataFrame(
            {c: [1.5, 2.0, 0.0] for c in columns}, index=index)
        rainfall = pd.Series([0.0, 12.5, 3.0], index=index, name=4)
        return loads, rainfall

    @staticmethod
    def _header(path):
        with open(path, encoding='utf-8') as f:
            return f.readline().rstrip('\n').split(',')

    def test_each_header_carries_its_units(self, tmp_path):
        loads, rainfall = self._replicate()
        tss, rain = tmp_path / 'fire_tss.csv', tmp_path / 'rain.csv'
        write_replicate_csvs(loads, rainfall, tss, rain, 'day')

        assert self._header(tss) == ['Date', 'SC1 (kg/day)', 'SC2 (kg/day)']
        assert self._header(rain) == ['Date', 'rainfall (mm/day)']

    def test_the_units_follow_the_timestep(self, tmp_path):
        loads, rainfall = self._replicate()
        tss, rain = tmp_path / 'fire_tss.csv', tmp_path / 'rain.csv'
        write_replicate_csvs(loads, rainfall, tss, rain, 'hour')

        assert self._header(tss)[1:] == ['SC1 (kg/hour)', 'SC2 (kg/hour)']
        assert self._header(rain)[1:] == ['rainfall (mm/hour)']

    def test_the_values_and_dates_are_written_unchanged(self, tmp_path):
        loads, rainfall = self._replicate()
        tss, rain = tmp_path / 'fire_tss.csv', tmp_path / 'rain.csv'
        write_replicate_csvs(loads, rainfall, tss, rain, 'day')

        back = pd.read_csv(tss, index_col=0, parse_dates=True)
        assert list(back.index) == list(loads.index)
        assert back.to_numpy().tolist() == loads.to_numpy().tolist()
        rain_back = pd.read_csv(rain, index_col=0, parse_dates=True)
        assert rain_back.iloc[:, 0].tolist() == rainfall.tolist()

    def test_every_timestep_writes_units_source_reads_from_a_header(self):
        """Checked on Source 6.10: 'kg/day', 'mm/day', 'kg/hour' and
        'mm/hour' load from a header with their units. A spelling Source
        does not know - 'kg/hr', say - is not an error: the column loads
        named 'X (kg/hr)', with no units, and nothing matches it. So a new
        timestep's unit word has to be checked against Source first."""
        checked_on_source = {'day', 'hour'}
        assert {t['units'] for t in SOURCE_TIMESTEPS.values()} <= (
            checked_on_source)

    def test_the_callers_frame_is_left_as_it_was(self, tmp_path):
        loads, rainfall = self._replicate()
        write_replicate_csvs(
            loads, rainfall, tmp_path / 'a.csv', tmp_path / 'b.csv', 'day')
        assert list(loads.columns) == ['SC1', 'SC2']


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

    def test_an_exact_match_wins_over_one_differing_in_case(self):
        v = FakeConstituents(['tss', 'TSS', 'Tss'])
        assert ensure_constituent(v, 'TSS') == 'TSS'
        assert v.added == []

    def test_several_matches_differing_only_in_case_are_refused(self):
        """Neither is obviously the one meant, and creating a third would
        only add to the confusion."""
        v = FakeConstituents(['Salt', 'tss', 'Tss'])
        with pytest.raises(ValueError) as raised:
            ensure_constituent(v, 'TSS')
        assert "'tss'" in str(raised.value) and "'Tss'" in str(raised.value)
        assert 'source.constituent' in str(raised.value)
        assert v.added == []

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

    @pytest.mark.parametrize('model', [
        ['Pollutant_N', 'Salt'], ['Contaminant'], ['Nitrogen Pollutant']])
    def test_left_unnamed_only_sediment_is_looked_for(self, model, caplog):
        """detect_constituent's candidates include 'Contaminant' and
        'Pollutant', which as often name a nutrient - and the Load
        Distributor step would replace its generation models."""
        v = FakeConstituents(model)
        assert ensure_constituent(v) == 'TSS'
        assert v.added == ['TSS']
        assert model[0] in caplog.text

    @pytest.mark.parametrize('name', [
        'TSS', 'tss_load', 'Fine Sediment', 'Total Suspended Solids'])
    def test_left_unnamed_a_sediment_constituent_is_found(self, name):
        v = FakeConstituents(['Nitrogen', name])
        assert ensure_constituent(v) == name
        assert v.added == []

    def test_a_constituent_source_did_not_create_is_reported(self):
        v = FakeConstituents([], adds=False)
        with pytest.raises(RuntimeError, match='TSS'):
            ensure_constituent(v, 'TSS')


class TestSaveModel:
    """save_model explains a Veneer that gives scripts no project handler."""

    OLD_VENEER_PY = "{'Message': \"'NoneType' object has no attribute 'CallBackHandler'\"}"
    NEW_VENEER_PY = ("{'Message': 'Cannot save: this Veneer server gave the script no "
                     "project handler (project_handler is None), ... Server process: "
                     "C:\\Somewhere\\Host.exe (process 1234)'}")

    def _veneer(self, error=None):
        v = MagicMock()
        v.port = 9880
        if error is not None:
            v.model.save.side_effect = Exception(error)
        return v

    def test_it_saves_to_the_filename(self):
        v = self._veneer()
        save_model(v, 'out.rsproj')
        v.model.save.assert_called_once_with('out.rsproj')

    @pytest.mark.parametrize('error', [OLD_VENEER_PY, NEW_VENEER_PY])
    def test_a_missing_project_handler_is_explained(self, error):
        with pytest.raises(RuntimeError) as err:
            save_model(self._veneer(error), 'out.rsproj')
        message = str(err.value)
        assert 'out.rsproj' in message
        assert '9880' in message
        assert 'project handler' in message
        assert err.value.__cause__ is not None

    def test_the_server_process_is_passed_on(self):
        with pytest.raises(RuntimeError, match='Host.exe'):
            save_model(self._veneer(self.NEW_VENEER_PY), 'out.rsproj')

    def test_any_other_failure_is_raised_unchanged(self):
        with pytest.raises(Exception, match='disk full') as err:
            save_model(self._veneer('disk full'), 'out.rsproj')
        assert not isinstance(err.value, RuntimeError)
