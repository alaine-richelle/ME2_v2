import json

import pytest

from me2.weather import (
    FORECAST_ENDPOINT,
    GEOCODING_ENDPOINT,
    WeatherServiceError,
    fetch_current_weather,
)


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self, size):
        return self.payload[:size]


def test_current_weather_uses_configured_city_and_parses_fresh_observation(monkeypatch):
    seen = []
    responses = iter([
        {
            "results": [{
                "name": "Manila", "admin1": "Metro Manila", "country": "Philippines",
                "latitude": 14.59, "longitude": 120.98,
            }],
        },
        {
            "timezone": "Asia/Manila",
            "current": {
                "time": "2026-10-14T12:00", "temperature_2m": 29.0,
                "relative_humidity_2m": 78, "apparent_temperature": 34.0,
                "precipitation": 0.0, "weather_code": 2, "wind_speed_10m": 4.0,
            },
            "current_units": {
                "temperature_2m": "°C", "apparent_temperature": "°C",
                "relative_humidity_2m": "%", "precipitation": "mm",
                "wind_speed_10m": "km/h",
            },
        },
    ])

    def fake_urlopen(request, timeout):
        seen.append((request.full_url, timeout))
        return FakeResponse(next(responses))

    monkeypatch.setattr("me2.weather.urlopen", fake_urlopen)
    result = fetch_current_weather("Manila", timeout_seconds=1.5)
    assert result["city"] == "Manila"
    assert result["condition"] == "Partly cloudy"
    assert result["temperature"] == 29
    assert result["observation_time"] == "2026-10-14T12:00"
    assert len(seen) == 2
    assert seen[0][0].startswith(GEOCODING_ENDPOINT)
    assert "name=Manila" in seen[0][0]
    assert seen[1][0].startswith(FORECAST_ENDPOINT)
    assert "timezone=auto" in seen[1][0]
    assert all(timeout == 1.5 for _, timeout in seen)


def test_unknown_city_and_invalid_weather_payload_fail_cleanly(monkeypatch):
    monkeypatch.setattr("me2.weather.urlopen", lambda *args, **kwargs: FakeResponse({"results": []}))
    with pytest.raises(WeatherServiceError, match="city was not found"):
        fetch_current_weather("No Such City")

    monkeypatch.setattr(
        "me2.weather.urlopen",
        lambda *args, **kwargs: FakeResponse({"results": [{
            "name": "Manila", "latitude": 14.59, "longitude": 120.98,
        }]}),
    )
    with pytest.raises(WeatherServiceError, match="no current observation"):
        fetch_current_weather("Manila")


def test_weather_timeout_and_city_validation():
    with pytest.raises(ValueError, match="city"):
        fetch_current_weather("")
    with pytest.raises(ValueError, match="timeout"):
        fetch_current_weather("Manila", timeout_seconds=0)


def test_weather_rejects_non_finite_service_values(monkeypatch):
    responses = iter([
        {"results": [{"name": "Manila", "latitude": 14.59, "longitude": 120.98}]},
        {
            "timezone": "Asia/Manila", "current": {
                "time": "2026-10-14T12:00", "temperature_2m": float("nan"),
                "relative_humidity_2m": 78, "apparent_temperature": 34,
                "precipitation": 0, "weather_code": 0, "wind_speed_10m": 4,
            }, "current_units": {},
        },
    ])
    monkeypatch.setattr(
        "me2.weather.urlopen",
        lambda *args, **kwargs: FakeResponse(next(responses)),
    )
    with pytest.raises(WeatherServiceError, match="non-finite"):
        fetch_current_weather("Manila")
