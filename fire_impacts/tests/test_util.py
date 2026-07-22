"""
General helpers: API retry, file discovery, parameter validation.

retry wraps every remote data call in the package, so its give-up
behaviour decides whether a flaky DEA/TERN request fails the run.
"""

import time

import pytest

from fire_impacts import util


@pytest.fixture()
def no_sleeping(monkeypatch):
    """
    Record back-off delays instead of waiting them out.

    retry does `import time` inside the function body, so it picks the
    module up from sys.modules on each call - patching time.sleep there
    is what it actually sees.
    """
    slept = []
    monkeypatch.setattr(time, 'sleep', slept.append)
    return slept


class Flaky:
    """Callable that fails a set number of times, then succeeds."""

    def __init__(self, failures, exception=RuntimeError):
        self.failures = failures
        self.exception = exception
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exception(f'attempt {self.calls}')
        return 'ok'


class TestRetry:

    def test_returns_immediately_on_success(self, no_sleeping):
        fn = Flaky(failures=0)

        assert util.retry(fn) == 'ok'
        assert fn.calls == 1
        assert no_sleeping == []

    def test_retries_until_it_succeeds(self, no_sleeping):
        fn = Flaky(failures=3)

        assert util.retry(fn, retries=5, initial_delay=1) == 'ok'
        assert fn.calls == 4

    def test_gives_up_and_reraises_the_last_error(self, no_sleeping):
        fn = Flaky(failures=99)

        with pytest.raises(RuntimeError, match='attempt 4'):
            util.retry(fn, retries=3, initial_delay=1)
        # One initial attempt plus three retries.
        assert fn.calls == 4

    def test_zero_retries_attempts_once(self, no_sleeping):
        fn = Flaky(failures=99)

        with pytest.raises(RuntimeError):
            util.retry(fn, retries=0)
        assert fn.calls == 1

    def test_backs_off_exponentially(self, no_sleeping):
        util_fn = Flaky(failures=99)

        with pytest.raises(RuntimeError):
            util.retry(util_fn, retries=3, initial_delay=2, delay_scale=3)

        assert no_sleeping == [2, 6, 18]

    def test_retries_listed_exceptions(self, no_sleeping):
        fn = Flaky(failures=1, exception=TimeoutError)

        result = util.retry(
            fn, retries=3, initial_delay=1,
            specific_exceptions=[TimeoutError],
        )
        assert result == 'ok'
        assert fn.calls == 2

    def test_reraises_unlisted_exceptions_without_retrying(self, no_sleeping):
        fn = Flaky(failures=99, exception=ValueError)

        with pytest.raises(ValueError):
            util.retry(
                fn, retries=3, initial_delay=1,
                specific_exceptions=[TimeoutError],
            )
        assert fn.calls == 1
        assert no_sleeping == []

    def test_subclasses_of_listed_exceptions_are_not_retried(self, no_sleeping):
        # KNOWN SHARP EDGE, pinned rather than endorsed: the check is
        # `e.__class__ not in specific_exceptions`, an exact identity
        # test rather than isinstance. ConnectionError is an OSError, but
        # listing OSError will not catch it - which is the wrong way
        # round for network code, where the concrete subclass is what
        # gets raised. No caller passes specific_exceptions today.
        fn = Flaky(failures=1, exception=ConnectionError)

        with pytest.raises(ConnectionError):
            util.retry(
                fn, retries=3, initial_delay=1,
                specific_exceptions=[OSError],
            )
        assert fn.calls == 1


class TestFileMatchingAll:

    @pytest.fixture()
    def files(self, tmp_path):
        for name in ('DEM_catch_a.tif', 'DEM_catch_b.tif',
                     'Slope_catch_a.tif', 'notes.txt'):
            (tmp_path / name).touch()
        return tmp_path

    def test_matches_a_single_substring(self, files):
        assert sorted(util.file_matching_all(files, 'DEM')) == \
            ['DEM_catch_a.tif', 'DEM_catch_b.tif']

    def test_requires_every_substring(self, files):
        assert util.file_matching_all(files, 'DEM', 'catch_a') == \
            ['DEM_catch_a.tif']

    def test_returns_names_not_paths(self, files):
        for name in util.file_matching_all(files, 'DEM'):
            assert '/' not in name

    def test_no_match_returns_empty(self, files):
        assert util.file_matching_all(files, 'Aridity') == []


class TestUniqueFileMatching:

    @pytest.fixture()
    def files(self, tmp_path):
        for name in ('DEM_catch_a.tif', 'DEM_catch_b.tif',
                     'Slope_catch_a.tif', 'Slope_catch_a.tfw'):
            (tmp_path / name).touch()
        return tmp_path

    def test_returns_the_single_match(self, files):
        assert util.unique_file_matching(files, 'DEM', 'catch_b') == \
            'DEM_catch_b.tif'

    def test_extension_narrows_an_otherwise_ambiguous_match(self, files):
        assert util.unique_file_matching(
            files, 'Slope', extension='.tif') == 'Slope_catch_a.tif'

    def test_no_match_raises(self, files):
        with pytest.raises(FileNotFoundError, match='No file found'):
            util.unique_file_matching(files, 'Aridity')

    def test_ambiguous_match_raises(self, files):
        # Silently picking one would make the choice depend on directory
        # ordering.
        with pytest.raises(FileExistsError, match='Multiple files found'):
            util.unique_file_matching(files, 'DEM')

    def test_extension_that_matches_nothing_raises(self, files):
        with pytest.raises(FileNotFoundError):
            util.unique_file_matching(files, 'DEM', extension='.nc')


class TestCheckAcceptableParam:

    def test_returns_the_normalised_value(self):
        assert util.check_acceptable_param('depth', ['depth', 'intensity']) \
            == 'depth'

    @pytest.mark.parametrize('given', ['DEPTH', ' depth ', 'Depth'])
    def test_normalises_case_and_whitespace(self, given):
        assert util.check_acceptable_param(given, ['depth']) == 'depth'

    def test_rejects_an_unacceptable_value(self):
        with pytest.raises(ValueError, match='must be one of'):
            util.check_acceptable_param('volume', ['depth', 'intensity'])


class TestDateRel:

    def test_adds_days(self):
        assert util.date_rel('2020-01-15', 10) == '2020-01-25'

    def test_subtracts_days(self):
        assert util.date_rel('2020-01-15', -20) == '2019-12-26'

    def test_crosses_a_leap_day(self):
        assert util.date_rel('2020-02-28', 2) == '2020-03-01'

    def test_zero_is_a_no_op(self):
        assert util.date_rel('2020-01-15', 0) == '2020-01-15'
