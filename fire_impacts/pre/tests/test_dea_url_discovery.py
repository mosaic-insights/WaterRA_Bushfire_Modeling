"""
Discovery of the latest available DEA Land Cover mosaic.

requests.head is mocked throughout - no network. The logic worth testing
is the distinction the function draws between "this year isn't published
yet" (try an older one) and "the server is unreachable or misconfigured"
(fail now, because every year uses the same host).
"""

import pytest
import requests

from fire_impacts.pre import mask_dnbr
from fire_impacts.pre.mask_dnbr import _find_latest_dea_url


class Response:
    def __init__(self, status_code):
        self.status_code = status_code


@pytest.fixture()
def head(monkeypatch):
    """Drive requests.head from a per-year status code map."""
    calls = []

    def configure(status_by_year, default=404):
        def fake_head(url, timeout=None, allow_redirects=None):
            year = int(url.rsplit('--P1Y_', 1)[0].rsplit('_', 1)[1])
            calls.append(year)
            outcome = status_by_year.get(year, default)
            if isinstance(outcome, Exception):
                raise outcome
            return Response(outcome)

        monkeypatch.setattr(mask_dnbr.requests, 'head', fake_head)
        return calls

    configure.calls = calls
    return configure


class TestYearFallback:

    def test_returns_the_first_available_year(self, head):
        calls = head({2024: 200})

        year, url = _find_latest_dea_url('level3', 2024, lookback=5)

        assert year == 2024
        assert '2024' in url
        assert calls == [2024]

    def test_falls_back_over_missing_years(self, head):
        calls = head({2024: 404, 2023: 404, 2022: 200})

        year, _ = _find_latest_dea_url('level3', 2024, lookback=5)

        assert year == 2022
        assert calls == [2024, 2023, 2022]

    def test_walks_backwards_from_the_start_year(self, head):
        calls = head({2020: 200})

        _find_latest_dea_url('level3', 2024, lookback=10)

        assert calls == [2024, 2023, 2022, 2021, 2020]

    def test_respects_the_lookback_limit(self, head):
        calls = head({}, default=404)

        with pytest.raises(RuntimeError, match='not found for any year'):
            _find_latest_dea_url('level3', 2024, lookback=2)

        # start_year plus lookback years.
        assert calls == [2024, 2023, 2022]

    def test_url_carries_the_requested_level(self, head):
        head({2024: 200})

        _, url = _find_latest_dea_url('level4', 2024, lookback=1)

        assert url.endswith('level4.tif')

    def test_default_start_year_is_last_year(self, head, monkeypatch):
        calls = head({}, default=200)

        year, _ = _find_latest_dea_url('level3', None, lookback=1)

        # Whatever "now" is, it must be the year before it, and it must
        # be the first thing tried.
        from datetime import datetime
        assert year == datetime.now().year - 1
        assert calls[0] == year


class TestFailFast:

    def test_connection_error_does_not_try_other_years(self, head):
        # Every year resolves to the same host, so walking back is
        # pointless once the host is unreachable.
        calls = head({2024: requests.ConnectionError('no route to host')})

        with pytest.raises(RuntimeError, match='Could not connect'):
            _find_latest_dea_url('level3', 2024, lookback=5)

        assert calls == [2024]

    def test_timeout_does_not_try_other_years(self, head):
        calls = head({2024: requests.Timeout('too slow')})

        with pytest.raises(RuntimeError, match='timed out'):
            _find_latest_dea_url('level3', 2024, lookback=5)

        assert calls == [2024]

    def test_other_request_errors_fail_immediately(self, head):
        calls = head({2024: requests.TooManyRedirects('loop')})

        with pytest.raises(RuntimeError, match='Unexpected network error'):
            _find_latest_dea_url('level3', 2024, lookback=5)

        assert calls == [2024]

    @pytest.mark.parametrize('status', [401, 403])
    def test_access_denied_is_not_treated_as_missing(self, head, status):
        calls = head({2024: status})

        with pytest.raises(RuntimeError, match='Access denied'):
            _find_latest_dea_url('level3', 2024, lookback=5)

        assert calls == [2024]

    @pytest.mark.parametrize('status', [500, 503, 302])
    def test_unexpected_status_codes_fail_immediately(self, head, status):
        calls = head({2024: status})

        with pytest.raises(RuntimeError, match='Unexpected HTTP'):
            _find_latest_dea_url('level3', 2024, lookback=5)

        assert calls == [2024]

    def test_a_late_server_error_still_fails(self, head):
        # 404s walk back as normal, but a 500 partway through stops it.
        calls = head({2024: 404, 2023: 500})

        with pytest.raises(RuntimeError, match='Unexpected HTTP'):
            _find_latest_dea_url('level3', 2024, lookback=5)

        assert calls == [2024, 2023]


class TestErrorMessages:

    def test_all_404_message_names_the_year_range(self, head):
        head({}, default=404)

        with pytest.raises(RuntimeError) as excinfo:
            _find_latest_dea_url('level3', 2024, lookback=3)

        message = str(excinfo.value)
        assert '2024' in message and '2021' in message

    def test_connection_error_is_chained(self, head):
        original = requests.ConnectionError('no route to host')
        head({2024: original})

        with pytest.raises(RuntimeError) as excinfo:
            _find_latest_dea_url('level3', 2024, lookback=1)

        assert excinfo.value.__cause__ is original
