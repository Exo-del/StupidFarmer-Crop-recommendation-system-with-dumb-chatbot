import json
import urllib.request
import urllib.error

REGION_COORDS = {
    "Africa": (6.5244, 3.3792),
    "Asia": (28.6139, 77.2090),
    "Europe": (52.5200, 13.4050),
    "N.America": (39.0997, -94.5786),
    "S.America": (-23.5505, -46.6333),
    "Australia": (-33.8688, 151.2093),
}

REGION_CITIES = {
    "Africa": "Lagos, Nigeria",
    "Asia": "New Delhi, India",
    "Europe": "Berlin, Germany",
    "N.America": "Kansas City, USA",
    "S.America": "São Paulo, Brazil",
    "Australia": "Sydney, Australia",
}


def fetch_weather(region: str) -> dict | None:
    lat, lng = REGION_COORDS[region]
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lng}&"
        f"current=temperature_2m,relative_humidity_2m"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "CropRec/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as e:
        print(f"Weather fetch error: {e}")
        return None

    try:
        current = data["current"]
        return {
            "temperature": current["temperature_2m"],
            "humidity": current["relative_humidity_2m"],
            "city": REGION_CITIES.get(region, region),
        }
    except KeyError as e:
        print(f"Weather data parse error: {e}")
        return None
