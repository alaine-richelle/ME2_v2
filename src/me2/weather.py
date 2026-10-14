"""Small, timeout-bounded Open-Meteo client used only by the local dashboard."""

from __future__ import annotations

import json
import math
from urllib.parse import urlencode
from urllib.request import Request, urlopen

GEOCODING_ENDPOINT = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
WEATHER_CODES = {
    0: "Clear",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Dense drizzle",
    56: "Freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Heavy freezing rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light rain showers",
    81: "Rain showers",
    82: "Heavy rain showers",
    85: "Snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}
CURRENT_FIELDS = (
    "temperature_2m,relative_humidity_2m,apparent_temperature,"
    "precipitation,weather_code,wind_speed_10m"
)


class WeatherServiceError(RuntimeError):
    """A safe, user-displayable weather lookup failure."""


def _get_json(url: str, timeout_seconds: float) -> dict:
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "ME2-local-dashboard/1"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            if response.status != 200:
                raise WeatherServiceError("weather service returned a non-success status")
            payload = response.read(1_000_001)
    except WeatherServiceError:
        raise
    except Exception as error:
        raise WeatherServiceError("weather service request failed") from error
    if len(payload) > 1_000_000:
        raise WeatherServiceError("weather service response exceeded the size limit")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WeatherServiceError("weather service returned invalid JSON") from error
    if not isinstance(value, dict):
        raise WeatherServiceError("weather service returned an invalid response")
    return value


def _finite_number(payload: dict, name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WeatherServiceError("weather response is missing a numeric observation")
    value = float(value)
    if not math.isfinite(value):
        raise WeatherServiceError("weather response contains a non-finite observation")
    return value


def fetch_current_weather(city: str, *, timeout_seconds: float = 4.0) -> dict:
    """Fetch one current observation for a configured city; no credential required."""

    if not isinstance(city, str) or not city.strip():
        raise ValueError("city must be a non-empty string")
    if not 0.2 <= float(timeout_seconds) <= 15:
        raise ValueError("timeout_seconds must be in [0.2, 15]")
    geocode_query = urlencode({"name": city.strip(), "count": 1, "language": "en", "format": "json"})
    locations = _get_json(f"{GEOCODING_ENDPOINT}?{geocode_query}", timeout_seconds)
    matches = locations.get("results")
    if not isinstance(matches, list) or not matches or not isinstance(matches[0], dict):
        raise WeatherServiceError("configured city was not found")
    location = matches[0]
    latitude = _finite_number(location, "latitude")
    longitude = _finite_number(location, "longitude")
    query = urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": CURRENT_FIELDS,
        "timezone": "auto",
    })
    response = _get_json(f"{FORECAST_ENDPOINT}?{query}", timeout_seconds)
    current = response.get("current")
    units = response.get("current_units")
    if not isinstance(current, dict) or not isinstance(units, dict):
        raise WeatherServiceError("weather response has no current observation")
    code_value = current.get("weather_code")
    if isinstance(code_value, bool) or not isinstance(code_value, int):
        raise WeatherServiceError("weather response has no valid condition code")
    observation_time = current.get("time")
    if not isinstance(observation_time, str) or not observation_time:
        raise WeatherServiceError("weather response has no observation time")
    return {
        "city": location.get("name", city.strip()),
        "region": location.get("admin1"),
        "country": location.get("country"),
        "observation_time": observation_time,
        "timezone": response.get("timezone"),
        "temperature": _finite_number(current, "temperature_2m"),
        "temperature_unit": units.get("temperature_2m", "°C"),
        "apparent_temperature": _finite_number(current, "apparent_temperature"),
        "relative_humidity_percent": _finite_number(current, "relative_humidity_2m"),
        "precipitation": _finite_number(current, "precipitation"),
        "precipitation_unit": units.get("precipitation", "mm"),
        "wind_speed": _finite_number(current, "wind_speed_10m"),
        "wind_speed_unit": units.get("wind_speed_10m", "km/h"),
        "weather_code": code_value,
        "condition": WEATHER_CODES.get(code_value, "Unknown conditions"),
    }
