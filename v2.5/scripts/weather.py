import json
import urllib.request
import urllib.error

REGION_COORDS = {
    "Africa": (6.5244, 3.3792),
    "Asia": (28.6139, 77.2090),
    "Europe": (52.5200, 13.4050),
    "North America": (39.0997, -94.5786),
    "South America": (-23.5505, -46.6333),
    "Global": (39.0997, -94.5786),
    "North Africa": (30.0444, 31.2357),
    "West Africa": (6.5244, 3.3792),
    "East Africa": (-1.2864, 36.8172),
    "Central Africa": (4.3500, 18.3833),
    "Southern Africa": (-25.7461, 28.1881),
}

REGION_CITIES = {
    "Africa": "Lagos, Nigeria",
    "Asia": "New Delhi, India",
    "Europe": "Berlin, Germany",
    "North America": "Kansas City, USA",
    "South America": "São Paulo, Brazil",
    "Global": "Kansas City, USA",
    "North Africa": "Cairo, Egypt",
    "West Africa": "Lagos, Nigeria",
    "East Africa": "Nairobi, Kenya",
    "Central Africa": "Bangui, CAR",
    "Southern Africa": "Pretoria, South Africa",
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
