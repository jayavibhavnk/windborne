import requests
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import Point

class CountryWeatherMonitor():
    """
    Real-Time Country Weather Monitoring Tool

    Parameters:
    - country_name: Name of target country (must exist in Natural Earth dataset)
    - buffer_km: Monitoring radius from border in kilometers (default 500)
    - plot: Generate visualization (default True)

    Returns:
    - summary: Monitoring operation summary
    - plot_path: Path to generated visualization
    - balloons_used: List of balloons in monitoring zone
    """

    def __init__(self):
        self.world = gpd.read_file(
            '110m_cultural.zip',
            layer='ne_110m_admin_0_countries'  # Explicit layer to prevent warnings
        )
        self.country_name = None
        self.buffer_km = None
        self.country_polygon = None
        self.buffered_polygon = None
        self.balloons = []

    def monitor(self, country_name, buffer_km=500, plot=True):
        """Main monitoring method with error handling"""
        result = {"summary": "", "plot_path": None, "balloons_used": []}

        try:
            # Original country setup logic
            country = self.world[self.world.NAME == country_name]
            if country.empty:
                raise ValueError(f"Country '{country_name}' not found")
            self.country_polygon = country.geometry.iloc[0]
            self.buffered_polygon = self.country_polygon.buffer(buffer_km/111)
            self.country_name = country_name
            self.buffer_km = buffer_km

            # Original balloon loading logic
            self.balloons = self._fetch_balloons()
            self._filter_balloons()

            # Generate outputs
            result["summary"] = self._generate_summary()
            result["balloons_used"] = self.balloons

            if plot:
                result["plot_path"] = self._generate_plot()

        except Exception as e:
            result["summary"] = f"Error: {str(e)}"

        return result

    def _fetch_balloons(self):
        """Original load_balloon_data logic"""
        try:
            response = requests.get("https://a.windbornesystems.com/treasure/00.json", timeout=10)
            return [{'lat': p[0], 'lon': p[1], 'alt': p[2]} for p in response.json() if len(p) == 3]
        except Exception as e:
            print(f"Balloon API error: {e}")
            return []

    def _filter_balloons(self):
        """Original filter_balloons logic"""
        relevant = []
        for balloon in self.balloons:
            point = Point(balloon['lon'], balloon['lat'])
            if self.buffered_polygon.contains(point):
                balloon['distance'] = self.country_polygon.distance(point) * 111
                relevant.append(balloon)
        self.balloons = sorted(relevant, key=lambda x: x['distance'])

    def _get_weather_data(self, lat, lon):
        """Original get_weather_data logic"""
        try:
            response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={'latitude': lat, 'longitude': lon,
                       'current': 'temperature_2m,wind_speed_10m,precipitation'},
                timeout=5
            )
            return response.json().get('current', {})
        except Exception as e:
            print(f"Weather API error: {e}")
            return None

    def _generate_plot(self):
        """Adapted plot_weather_map with auto-saving"""
        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Base map elements
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.COASTLINE)

        # Plot boundaries
        gpd.GeoSeries([self.country_polygon]).boundary.plot(
            ax=ax, color='red', linewidth=2, label='Country Border'
        )
        gpd.GeoSeries([self.buffered_polygon]).boundary.plot(
            ax=ax, color='blue', linestyle='--', linewidth=1, label='Monitoring Zone'
        )

        # Plot balloons
        for balloon in self.balloons:
            weather = self._get_weather_data(balloon['lat'], balloon['lon'])
            color = self._temp_color(weather.get('temperature_2m', 20))

            ax.plot(balloon['lon'], balloon['lat'], 'o',
                    color=color, markersize=12, alpha=0.7,
                    transform=ccrs.Geodetic())

            if weather:
                text = (f"Temp: {weather['temperature_2m']}°C\n"
                        f"Wind: {weather['wind_speed_10m']} km/h\n"
                        f"Precip: {weather['precipitation']} mm")
                ax.text(balloon['lon'] + 0.3, balloon['lat'], text,
                        fontsize=8, bbox=dict(facecolor='white', alpha=0.8),
                        transform=ccrs.Geodetic())

        plt.legend()
        plt.title(f"Real-Time Weathera Monitoring for {self.country_name}\nUsing {len(self.balloons)} Balloons")

        plot_path = f"weather_monitor_{datetime.now().strftime('%Y%m%d%H%M')}.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def _temp_color(self, temp):
        """Original _get_temp_color logic"""
        if temp < 0: return 'blue'
        if temp < 10: return 'lightblue'
        if temp < 20: return 'green'
        if temp < 30: return 'yellow'
        return 'red'

    def _generate_summary(self):
        """Generate monitoring summary"""
        return (f"Monitoring {self.country_name} ({self.buffer_km}km radius)\n"
               f"Balloons available: {len(self.balloons)}\n"
               f"Balloon positions: {[b['distance'] for b in self.balloons]}")

# # Usage example matching original functionality
# if __name__ == "__main__":
#     monitor = CountryWeatherMonitor()
#     result = monitor.monitor("India", buffer_km=300)
#     print(result["summary"])
#     if result["plot_path"]:
#         print(f"Plot saved to: {result['plot_path']}")

## tool 1
import requests
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class BalloonTracker:
    """
    Tool 1: Balloon Position Tracker

    Provides real-time and historical balloon positions visualization

    Parameters:
    - num_balloons: Number of balloons to display (default: all)
    - history_hours: Hours of historical data to show (default: 24)

    Outputs:
    - Matplotlib plot of positions
    - Text summary of coverage
    - Path to saved image
    """

    def __init__(self):
        self.base_url = "https://a.windbornesystems.com/treasure"
        self.last_fetched: Dict[int, List] = {}

    def fetch_historical_data(self, hours_ago: int) -> Optional[List]:
        """Fetch balloon positions from API with error handling"""
        try:
            url = f"{self.base_url}/{hours_ago:02d}.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching {hours_ago}h data: {str(e)[:80]}")
            return None

    def get_positions(self, num_balloons: int = None, history_hours: int = 24) -> Dict:
        """
        Get balloon positions with optional sampling

        Args:
            num_balloons: Number of balloons to return (None = all)
            history_hours: Hours of historical data to collect

        Returns:
            {
                'positions': {hour: [[lat, lon, alt], ...]},
                'summary': str,
                'plot_path': str
            }
        """
        positions = {}
        valid_hours = 0

        for hours_ago in range(history_hours):
            data = self.fetch_historical_data(hours_ago)
            if data:
                positions[hours_ago] = data[:num_balloons] if num_balloons else data
                valid_hours += 1

        return {
            'positions': positions,
            'summary': self.generate_summary(positions, valid_hours, history_hours),
            'plot_path': self.plot_positions(positions)
        }

    def plot_positions(self, positions: Dict[int, List]) -> str:
        """Generate visualization of balloon positions"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Add map features
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Plot positions
        for hour, data in positions.items():
            lons = [p[1] for p in data]
            lats = [p[0] for p in data]
            ax.scatter(lons, lats, s=10, alpha=0.7,
                      transform=ccrs.PlateCarree(),
                      label=f"{hour}h ago")

        ax.set_global()
        plt.legend(loc='lower left', bbox_to_anchor=(0.1, 0.1))
        plt.title(f"Balloon Positions - Last {len(positions)} Hours")

        # Save and return path
        filepath = f"balloon_positions_{datetime.now().strftime('%Y%m%d%H%M')}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        return filepath

    def generate_summary(self, positions: Dict, valid: int, requested: int) -> str:
        """Generate text summary for LLM consumption"""
        total = sum(len(v) for v in positions.values())
        latest = len(positions.get(0, []))
        coverage = valid/requested * 100 if requested > 0 else 0

        return (
            f"Balloon Position Summary:\n"
            f"- {valid}/{requested} hours of data available ({coverage:.1f}% coverage)\n"
            f"- {total} total position records\n"
            f"- {latest} balloons in current hour\n"
            f"- Spatial coverage: {self._get_geo_range(positions)}"
        )

    def _get_geo_range(self, positions: Dict) -> str:
        """Calculate geographic coverage description"""
        all_lons = [p[1] for data in positions.values() for p in data]
        all_lats = [p[0] for data in positions.values() for p in data]

        if not all_lats:
            return "No position data available"

        return (
            f"Latitude: {min(all_lats):.1f}° to {max(all_lats):.1f}°\n"
            f"Longitude: {min(all_lons):.1f}° to {max(all_lons):.1f}°"
        )

## tool 2
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.stats import linregress
from typing import Dict, List, Tuple
import requests

class AltitudeAnalyzer:
    """
    Tool 2: Atmospheric Altitude Analysis

    Analyzes balloon altitude patterns and their relationship to wind speeds

    Parameters:
    - num_hours: Historical data hours to analyze (default: 4)
    - max_altitude: Maximum valid altitude in km (default: 50)

    Outputs:
    - Three analysis plots
    - Text summary of altitude patterns
    - Wind shear statistics
    - Plot file paths
    """

    def __init__(self):
        self.base_url = "https://a.windbornesystems.com/treasure"
        self.atmospheric_layers = {
            'Troposphere': (0, 12),
            'Stratosphere': (12, 50),
            'Mesosphere': (50, 85)
        }

    def fetch_balloon_data(self, num_hours: int = 4) -> List[List]:
        """Fetch and validate historical balloon data"""
        positions = []
        for hour in range(num_hours):
            try:
                url = f"{self.base_url}/{hour:02d}.json"
                response = requests.get(url, timeout=10)
                data = response.json()
                positions.append([p for p in data if self._is_valid_entry(p)])
            except Exception as e:
                print(f"Error fetching {hour}h data: {str(e)[:50]}...")
                positions.append([])
        return positions

    def _is_valid_entry(self, entry: List) -> bool:
        """Validate balloon position entry"""
        return (len(entry) == 3 and
                -90 <= entry[0] <= 90 and
                -180 <= entry[1] <= 180 and
                0 <= entry[2] <= 50)

    def analyze_altitude(self, num_hours: int = 4) -> Dict:
        """
        Main analysis method

        Returns:
            {
                'summary': str,
                'plots': {
                    'scatter': str,
                    'distribution': str,
                    'boxplot': str
                },
                'stats': {
                    'wind_shear': float,
                    'layer_stats': Dict,
                    'data_points': int
                }
            }
        """
        positions = self.fetch_balloon_data(num_hours)
        wind_data = self._calculate_wind_vectors(positions)

        return {
            'summary': self._generate_summary(wind_data),
            'plots': self._generate_plots(wind_data),
            'stats': self._calculate_statistics(wind_data)
        }

    def _calculate_wind_vectors(self, positions: List[List]) -> List[Dict]:
        """Calculate wind vectors from position changes"""
        wind_data = []
        current = next((p for p in positions if p), [])

        for prev in positions[1:]:
            for i in range(min(len(current), len(prev))):
                try:
                    lat1, lon1, alt1 = current[i]
                    lat2, lon2, alt2 = prev[i]

                    dx = (lon1 - lon2) * 111 * np.cos(np.radians(lat1))
                    dy = (lat1 - lat2) * 111
                    speed = np.hypot(dx, dy)

                    wind_data.append({
                        'lat': lat1,
                        'lon': lon1,
                        'altitude': alt1,
                        'speed': speed,
                        'direction': np.degrees(np.arctan2(dx, dy)) % 360
                    })
                except:
                    continue
        return wind_data

    def _generate_plots(self, wind_data: List[Dict]) -> Dict:
        """Generate and save all analysis plots"""
        valid_data = [d for d in wind_data if 0 < d['speed'] < 300]
        timestamp = datetime.now().strftime("%Y%m%d%H%M")

        return {
            'scatter': self._plot_scatter(valid_data, timestamp),
            'distribution': self._plot_distribution(valid_data, timestamp),
            'boxplot': self._plot_boxplots(valid_data, timestamp)
        }

    def _plot_scatter(self, data: List[Dict], timestamp: str) -> str:
        """Altitude vs wind speed with regression"""
        fig = plt.figure(figsize=(8, 6))
        altitudes = [d['altitude'] for d in data]
        speeds = [d['speed'] for d in data]

        slope, _, r_value, _, _ = linregress(altitudes, speeds)
        plt.scatter(altitudes, speeds, c=altitudes, cmap='viridis', alpha=0.6)
        plt.plot(altitudes, slope*np.array(altitudes), 'r--')
        plt.xlabel('Altitude (km)')
        plt.ylabel('Wind Speed (km/h)')
        plt.title(f'Altitude vs Wind Speed (R²={r_value**2:.2f})')
        path = f"altitude_scatter_{timestamp}.png"
        plt.savefig(path, dpi=120)
        plt.close()
        return path

    def _plot_distribution(self, data: List[Dict], timestamp: str) -> str:
        """Global altitude distribution map"""
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.coastlines()

        lons = [d['lon'] for d in data]
        lats = [d['lat'] for d in data]
        altitudes = [d['altitude'] for d in data]

        sc = ax.scatter(lons, lats, c=altitudes, cmap='plasma', s=20)
        plt.colorbar(sc, label='Altitude (km)')
        path = f"altitude_distribution_{timestamp}.png"
        plt.savefig(path, dpi=120)
        plt.close()
        return path

    def _plot_boxplots(self, data: List[Dict], timestamp: str) -> str:
        """Wind speed distribution by atmospheric layer"""
        layer_speeds = {name: [] for name in self.atmospheric_layers}
        for d in data:
            for name, (bottom, top) in self.atmospheric_layers.items():
                if bottom <= d['altitude'] < top:
                    layer_speeds[name].append(d['speed'])

        plt.figure(figsize=(10, 6))
        plt.boxplot(layer_speeds.values(), labels=layer_speeds.keys())
        plt.ylabel('Wind Speed (km/h)')
        plt.title('Wind Speed Distribution by Atmospheric Layer')
        path = f"altitude_boxplots_{timestamp}.png"
        plt.savefig(path, dpi=120)
        plt.close()
        return path

    def _generate_summary(self, data: List[Dict]) -> str:
        """Generate natural language summary"""
        if not data:
            return "No valid wind data available for analysis"

        stats = self._calculate_statistics(data)
        return (
            "Altitude Analysis Summary:\n"
            f"- Analyzed {stats['data_points']} balloon measurements\n"
            f"- Average wind shear: {stats['wind_shear']:.2f} (km/h)/km\n"
            "Layer Statistics:\n" +
            "\n".join([f"{layer}: {desc}" for layer, desc in stats['layer_stats'].items()])
        )

    def _calculate_statistics(self, data: List[Dict]) -> Dict:
        """Calculate numerical statistics"""
        if not data:
            return {}

        altitudes = [d['altitude'] for d in data]
        speeds = [d['speed'] for d in data]
        wind_shear = np.polyfit(altitudes, speeds, 1)[0]

        layer_stats = {}
        for name, (bottom, top) in self.atmospheric_layers.items():
            layer_data = [d['speed'] for d in data if bottom <= d['altitude'] < top]
            if layer_data:
                layer_stats[name] = (
                    f"n={len(layer_data)}, μ={np.mean(layer_data):.1f} km/h, "
                    f"σ={np.std(layer_data):.1f} km/h"
                )

        return {
            'wind_shear': wind_shear,
            'layer_stats': layer_stats,
            'data_points': len(data)
        }
import requests
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

class WindSpeedTool:
    def __init__(self):
        self.base_url = "https://a.windbornesystems.com/treasure"

    def analyze_winds(self, hours=2, top_n=5):
        """Original functionality packaged as-is with same outputs"""
        positions = self._fetch_data(hours)
        current, previous = self._get_positions(positions)
        balloon_winds = self._calculate_vectors(current, previous)

        return {
            "text_output": self._get_text_output(balloon_winds[:top_n]),
            "plot_path": self._generate_plot(balloon_winds),
            "raw_data": balloon_winds[:top_n]
        }

    # Original fetch_balloon_data converted to method
    def _fetch_data(self, hours):
        positions = []
        for hour in range(hours):
            try:
                url = f"{self.base_url}/{hour:02d}.json"
                response = requests.get(url, timeout=10)
                data = response.json()
                if isinstance(data, list) and len(data) >= 1000:
                    validated = [entry for entry in data if self._valid_entry(entry)]
                    positions.append(validated)
                else:
                    positions.append(None)
            except Exception as e:
                print(f"Balloon data error (hour {hour}): {str(e)[:50]}...")
                positions.append(None)
        return positions

    def _valid_entry(self, entry):
        return (len(entry) == 3 and
               -90 <= entry[0] <= 90 and
               -180 <= entry[1] <= 180)

    # Original main logic for current/previous positions
    def _get_positions(self, positions):
        current = next(p for p in positions if p is not None)
        previous = next(p for p in reversed(positions) if p is not None and p != current)
        return current, previous

    # Original calculate_wind_vector logic
    def _calculate_vectors(self, current, previous):
        balloon_winds = []
        for i in range(min(len(current), len(previous))):
            wind_vector = self._calculate_vector(previous[i], current[i])
            if wind_vector: balloon_winds.append(wind_vector)
        return balloon_winds

    def _calculate_vector(self, prev, current):
        try:
            lat1, lon1, alt1 = current
            lat2, lon2, alt2 = prev
            delta_lat = (lat1 - lat2) * 111
            delta_lon = (lon1 - lon2) * 111 * np.cos(np.radians(lat1))
            return {
                'speed': np.sqrt(delta_lat**2 + delta_lon**2),
                'position': (lat1, lon1, alt1)
            }
        except:
            return None

    # Original get_openmeteo_wind logic
    def _get_api_wind(self, lat, lon):
        try:
            params = {
                'latitude': lat,
                'longitude': lon,
                'hourly': 'wind_speed_10m',
                'forecast_days': 1
            }
            response = requests.get("https://api.open-meteo.com/v1/forecast",
                                  params=params, timeout=10)
            data = response.json()
            return next((s for s in data['hourly']['wind_speed_10m'] if s is not None), None)
        except:
            return None

    # Original plot_wind_analysis converted to save plot
    def _generate_plot(self, balloon_winds):
        valid_data = [w for w in balloon_winds if w and w['speed'] > 0]
        if not valid_data: return None

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        lons = [d['position'][1] for d in valid_data]
        lats = [d['position'][0] for d in valid_data]
        speeds = [d['speed'] for d in valid_data]

        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE, edgecolor='black')

        buffer = 2
        ax.set_extent([
            max(min(lons) - buffer, -180),
            min(max(lons) + buffer, 180),
            max(min(lats) - buffer, -90),
            min(max(lats) + buffer, 90)
        ])

        sc = ax.scatter(lons, lats, c=speeds, cmap='viridis', s=50,
                       transform=ccrs.PlateCarree(), edgecolor='black')
        plt.colorbar(sc, orientation='horizontal', pad=0.05)
        plt.title(f"Wind Speed Analysis\n{datetime.now().strftime('%Y-%m-%d %H:%M')}")

        plot_path = f"wind_speed_{datetime.now().strftime('%Y%m%d%H%M')}.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    # Original text output generation
    def _get_text_output(self, balloon_winds):
        output = ["Wind speed comparison:"]
        for i, wind in enumerate(balloon_winds):
            api_speed = self._get_api_wind(wind['position'][0], wind['position'][1])
            output.append(
                f"Balloon {i+1}: Calculated {wind['speed']:.1f} km/h | "
                f"API: {api_speed or 'N/A'} km/h"
            )
        return "\n".join(output)

## tool 4
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import requests
from datetime import datetime

class BalloonWeatherAnalyzer:
    """
    Tool 4: Top Balloon Weather Conditions

    Provides weather conditions around top N balloons with visualization

    Parameters:
    - n: Number of balloons to analyze (default: 5)
    - plot: Generate visualization (default: True)

    Outputs:
    - Summary of weather conditions
    - Path to generated plot
    - Raw weather data
    """

    def __init__(self):
        self.balloon_url = "https://a.windbornesystems.com/treasure/00.json"
        self.weather_url = "https://api.open-meteo.com/v1/forecast"

    def analyze(self, n=5, plot=True):
        """Main analysis method"""
        # Fetch and process data
        balloons = self._fetch_balloons(n)
        weather_data = self._get_weather_data(balloons)

        # Generate outputs
        return {
            "summary": self._generate_summary(weather_data),
            "plot_path": self._generate_plot(balloons, weather_data) if plot else None,
            "raw_data": weather_data
        }

    def _fetch_balloons(self, n):
        """Fetch top N balloons with error handling"""
        try:
            response = requests.get(self.balloon_url, timeout=10)
            raw_data = response.json()
            return [{
                'lat': entry[0],
                'lon': entry[1],
                'alt': entry[2]
            } for entry in raw_data[:n]]
        except Exception as e:
            print(f"Balloon API error: {e}")
            return []

    def _get_weather_data(self, balloons):
        """Get weather for each balloon position"""
        results = []
        for balloon in balloons:
            try:
                params = {
                    'latitude': balloon['lat'],
                    'longitude': balloon['lon'],
                    'current': 'temperature_2m,wind_speed_10m',
                    'hourly': 'relative_humidity_2m'
                }
                response = requests.get(self.weather_url, params=params, timeout=5)
                results.append(response.json())
            except:
                results.append(None)
        return results

    def _generate_plot(self, balloons, weather_data):
        """Generate and save visualization"""
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.stock_img()
        ax.coastlines()

        for balloon, weather in zip(balloons, weather_data):
            info_text = f"Altitude: {balloon['alt']:.1f} km"
            if weather and 'current' in weather:
                current = weather['current']
                info_text += f"\nTemp: {current['temperature_2m']}°C\nWind: {current['wind_speed_10m']} km/h"

            ax.plot(balloon['lon'], balloon['lat'], 'ro', markersize=10)
            ax.text(
                balloon['lon'] + 0.5, balloon['lat'],
                info_text,
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7)
            )

        plt.title(f'Top {len(balloons)} Balloons with Weather Conditions')
        plot_path = f"balloon_weather_{datetime.now().strftime('%Y%m%d%H%M')}.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def _generate_summary(self, weather_data):
        """Generate text summary"""
        summary = []
        for i, data in enumerate(weather_data):
            if data and 'current' in data:
                current = data['current']
                summary.append(
                    f"Balloon {i+1}: "
                    f"{current['temperature_2m']}°C, "
                    f"{current['wind_speed_10m']} km/h winds"
                )
        return "\n".join(summary) if summary else "No weather data available"

# Schema for LLM integration
TOOL4_SCHEMA = {
    "name": "balloon_weather_analysis",
    "description": "Analyzes weather conditions around top balloons",
    "parameters": {
        "type": "object",
        "properties": {
            "n": {
                "type": "integer",
                "description": "Number of balloons to analyze (default: 5)"
            },
            "plot": {
                "type": "boolean",
                "description": "Generate visualization (default: True)"
            }
        }
    },
    "returns": {
        "summary": "Text summary of weather conditions",
        "plot_path": "Path to generated visualization",
        "raw_data": "List of weather data dictionaries"
    }
}

## tool 5
import requests
from haversine import haversine
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from datetime import datetime

class BalloonPlaneProximity:
    """
    Tool 5: Balloon-Aircraft Proximity Checker

    Identifies balloons and aircraft in close proximity with visualization

    Parameters:
    - distance_km: Alert threshold in kilometers (default: 100)
    - plot: Generate visualization (default: True)

    Outputs:
    - Summary of nearby aircraft
    - Path to proximity plot
    - Raw proximity data
    """

    def __init__(self):
        self.balloon_url = "https://a.windbornesystems.com/treasure/00.json"
        self.opensky_url = "https://opensky-network.org/api/states/all"

    def analyze(self, distance_km=100, plot=True):
        """Main analysis method"""
        # Fetch and validate data
        balloons = self._fetch_balloons()
        planes = self._fetch_planes(balloons)

        # Calculate proximities
        results = self._calculate_proximities(balloons, planes, distance_km)

        return {
            "summary": self._generate_summary(results),
            "plot_path": self._generate_plot(results, balloons, planes) if plot else None,
            "raw_data": results
        }

    def _fetch_balloons(self):
        """Fetch balloon positions with validation"""
        try:
            response = requests.get(self.balloon_url, timeout=10)
            response.raise_for_status()
            raw_data = response.json()
            return [self._parse_balloon(pos) for pos in raw_data]
        except Exception as e:
            raise RuntimeError(f"Balloon data error: {str(e)}")

    def _parse_balloon(self, pos):
        """Validate balloon position format"""
        try:
            return {
                "lat": float(pos[0]),
                "lon": float(pos[1]),
                "alt": float(pos[2])
            }
        except (IndexError, ValueError, TypeError):
            return None

    def _fetch_planes(self, balloons):
        """Fetch plane data within balloon bounding box"""
        try:
            bbox = self._calculate_bbox(balloons)
            params = {
                "lamin": bbox["min_lat"],
                "lamax": bbox["max_lat"],
                "lomin": bbox["min_lon"],
                "lomax": bbox["max_lon"]
            }
            response = requests.get(self.opensky_url, params=params, timeout=15)
            response.raise_for_status()
            return response.json().get("states", [])
        except Exception as e:
            raise RuntimeError(f"Plane data error: {str(e)}")

    def _calculate_bbox(self, balloons, margin=0.5):
        """Calculate bounding box with safety margins"""
        valid_balloons = [b for b in balloons if b]
        lats = [b["lat"] for b in valid_balloons]
        lons = [b["lon"] for b in valid_balloons]

        return {
            "min_lat": max(min(lats) - margin, -90) if lats else -90,
            "max_lat": min(max(lats) + margin, 90) if lats else 90,
            "min_lon": max(min(lons) - margin, -180) if lons else -180,
            "max_lon": min(max(lons) + margin, 180) if lons else 180
        }

    def _calculate_proximities(self, balloons, planes, max_distance):
        """Calculate 3D distances between balloons and planes"""
        results = []
        for idx, balloon in enumerate(balloons):
            if not balloon: continue

            nearby = []
            for plane in planes:
                plane_data = self._parse_plane(plane)
                if not plane_data: continue

                distance = self._calculate_3d_distance(balloon, plane_data)
                if distance <= max_distance:
                    nearby.append(plane_data | {"distance_km": distance})

            if nearby:
                results.append({
                    "balloon_index": idx,
                    "balloon_data": balloon,
                    "nearby_planes": nearby
                })
        return results

    def _parse_plane(self, plane):
        """Extract and validate plane data"""
        try:
            return {
                "callsign": plane[1].strip(),
                "latitude": plane[6],
                "longitude": plane[5],
                "altitude": plane[13] or plane[7] or 0,
                "velocity": plane[9]
            }
        except (IndexError, TypeError):
            return None

    def _calculate_3d_distance(self, balloon, plane):
        """Calculate combined horizontal/vertical distance"""
        horizontal = haversine(
            (balloon["lat"], balloon["lon"]),
            (plane["latitude"], plane["longitude"])
        )
        vertical = abs(balloon["alt"] - plane["altitude"]) / 1000  # meters to km
        return (horizontal**2 + vertical**2)**0.5

    def _generate_plot(self, results, balloons, planes):
        """Generate and save proximity visualization"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Base map features
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Plot elements
        self._plot_balloons(ax, balloons, results)
        self._plot_planes(ax, planes)

        # Finalize plot
        ax.legend(loc='upper left')
        plot_path = f"proximity_{datetime.now().strftime('%Y%m%d%H%M')}.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def _plot_balloons(self, ax, balloons, results):
        """Plot balloon positions"""
        danger_indices = {r["balloon_index"] for r in results}
        safe_lons = [b["lon"] for i, b in enumerate(balloons) if b and i not in danger_indices]
        safe_lats = [b["lat"] for i, b in enumerate(balloons) if b and i not in danger_indices]
        danger_lons = [b["lon"] for r in results for b in [r["balloon_data"]]]
        danger_lats = [b["lat"] for r in results for b in [r["balloon_data"]]]

        ax.scatter(safe_lons, safe_lats, color='blue', s=5,
                   transform=ccrs.PlateCarree(), label='Safe Balloons')
        ax.scatter(danger_lons, danger_lats, color='red', s=20,
                   transform=ccrs.PlateCarree(), label='Proximity Balloons')

    def _plot_planes(self, ax, planes):
        """Plot aircraft positions"""
        plane_lons = [p[5] for p in planes if p[5] is not None]
        plane_lats = [p[6] for p in planes if p[6] is not None]
        ax.scatter(plane_lons, plane_lats, color='black', marker='^', s=5,
                   transform=ccrs.PlateCarree(), label='Aircraft')

    def _generate_summary(self, results):
        """Generate text report"""
        if not results:
            return "No balloons with nearby aircraft detected"

        summary = ["Balloons with nearby aircraft:"]
        for result in results:
            balloon = result["balloon_data"]
            summary.append(
                f"Balloon {result['balloon_index']} at "
                f"{balloon['lat']:.4f}°N, {balloon['lon']:.4f}°E "
                f"({len(result['nearby_planes'])} aircraft within range)"
            )
        return "\n".join(summary)

import requests
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import Point

class CountryWeatherMonitor:
    """
    Real-Time Country Weather Monitoring Tool

    Parameters:
    - country_name: Name of target country (must exist in Natural Earth dataset)
    - buffer_km: Monitoring radius from border in kilometers (default 500)
    - plot: Generate visualization (default True)

    Returns:
    - summary: Monitoring operation summary
    - plot_path: Path to generated visualization
    - balloons_used: List of balloons in monitoring zone
    """

    def __init__(self):
        self.world = gpd.read_file(
            '110m_cultural.zip',
            layer='ne_110m_admin_0_countries'  # Explicit layer to prevent warnings
        )

    def monitor(self, country_name, buffer_km=500, plot=True):
        """Main monitoring method with error handling"""
        result = {"summary": "", "plot_path": None, "balloons_used": []}

        try:
            # Original country setup logic
            country = self.world[self.world.NAME == country_name]
            if country.empty:
                raise ValueError(f"Country '{country_name}' not found")
            self.country_polygon = country.geometry.iloc[0]
            self.buffered_polygon = self.country_polygon.buffer(buffer_km/111)
            self.country_name = country_name
            self.buffer_km = buffer_km

            # Original balloon loading logic
            self.balloons = self._fetch_balloons()
            self._filter_balloons()

            # Generate outputs
            result["summary"] = self._generate_summary()
            result["balloons_used"] = self.balloons

            if plot:
                result["plot_path"] = self._generate_plot()

        except Exception as e:
            result["summary"] = f"Error: {str(e)}"

        return result

    def _fetch_balloons(self):
        """Original load_balloon_data logic"""
        try:
            response = requests.get("https://a.windbornesystems.com/treasure/00.json", timeout=10)
            return [{'lat': p[0], 'lon': p[1], 'alt': p[2]} for p in response.json() if len(p) == 3]
        except Exception as e:
            print(f"Balloon API error: {e}")
            return []

    def _filter_balloons(self):
        """Original filter_balloons logic"""
        relevant = []
        for balloon in self.balloons:
            point = Point(balloon['lon'], balloon['lat'])
            if self.buffered_polygon.contains(point):
                balloon['distance'] = self.country_polygon.distance(point) * 111
                relevant.append(balloon)
        self.balloons = sorted(relevant, key=lambda x: x['distance'])

    def _get_weather_data(self, lat, lon):
        """Original get_weather_data logic"""
        try:
            response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={'latitude': lat, 'longitude': lon,
                       'current': 'temperature_2m,wind_speed_10m,precipitation'},
                timeout=5
            )
            return response.json().get('current', {})
        except Exception as e:
            print(f"Weather API error: {e}")
            return None

    def _generate_plot(self):
        """Adapted plot_weather_map with auto-saving"""
        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Base map elements
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.COASTLINE)

        # Plot boundaries
        gpd.GeoSeries([self.country_polygon]).boundary.plot(
            ax=ax, color='red', linewidth=2, label='Country Border'
        )
        gpd.GeoSeries([self.buffered_polygon]).boundary.plot(
            ax=ax, color='blue', linestyle='--', linewidth=1, label='Monitoring Zone'
        )

        # Plot balloons
        for balloon in self.balloons:
            weather = self._get_weather_data(balloon['lat'], balloon['lon'])
            color = self._temp_color(weather.get('temperature_2m', 20))

            ax.plot(balloon['lon'], balloon['lat'], 'o',
                    color=color, markersize=12, alpha=0.7,
                    transform=ccrs.Geodetic())

            if weather:
                text = (f"Temp: {weather['temperature_2m']}°C\n"
                        f"Wind: {weather['wind_speed_10m']} km/h\n"
                        f"Precip: {weather['precipitation']} mm")
                ax.text(balloon['lon'] + 0.3, balloon['lat'], text,
                        fontsize=8, bbox=dict(facecolor='white', alpha=0.8),
                        transform=ccrs.Geodetic())

        plt.legend()
        plt.title(f"Real-Time Weather Monitoring for {self.country_name}\nUsing {len(self.balloons)} Balloons")

        plot_path = f"weather_monitor_{datetime.now().strftime('%Y%m%d%H%M')}.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def _temp_color(self, temp):
        """Original _get_temp_color logic"""
        if temp < 0: return 'blue'
        if temp < 10: return 'lightblue'
        if temp < 20: return 'green'
        if temp < 30: return 'yellow'
        return 'red'

    def _generate_summary(self):
        """Generate monitoring summary"""
        return (f"Monitoring {self.country_name} ({self.buffer_km}km radius)\n"
               f"Balloons available: {len(self.balloons)}\n"
               f"Balloon positions: {[b['distance'] for b in self.balloons]}")


import json
import re
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from langchain_groq import ChatGroq
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
import geopandas as gpd


# Wrapper Functions with Standardized Outputs
def balloon_tracker_wrapper(input_str):
    try:
        input_dict = json.loads(input_str)
        result = BalloonTracker().get_positions(**input_dict)
        summary = result['summary']
        plot_path = result['plot_path']
        return f"Task completed. Balloon positions tracked.\n{summary}\nPlot generated at: {plot_path}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON input for BalloonTracker"
    except Exception as e:
        return f"Error in BalloonTracker: {str(e)}"

def altitude_analyzer_wrapper(input_str):
    try:
        input_dict = json.loads(input_str)
        result = AltitudeAnalyzer().analyze_altitude(**input_dict)
        summary = result['summary']
        plots = "\n".join([f"{k}: {v}" for k, v in result['plots'].items()])
        return f"Task completed. Altitude analysis completed.\n{summary}\nPlots generated at:\n{plots}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON input for AltitudeAnalyzer"
    except Exception as e:
        return f"Error in AltitudeAnalyzer: {str(e)}"

def wind_speed_tool_wrapper(input_str):
    try:
        input_dict = json.loads(input_str)
        result = WindSpeedTool().analyze_winds(**input_dict)
        text_output = result['text_output']
        plot_path = result['plot_path']
        return f"Task completed. Wind speed analysis completed.\n{text_output}\nPlot generated at: {plot_path}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON input for WindSpeedTool"
    except Exception as e:
        return f"Error in WindSpeedTool: {str(e)}"

def balloon_weather_analyzer_wrapper(input_str):
    try:
        input_dict = json.loads(input_str)
        result = BalloonWeatherAnalyzer().analyze(**input_dict)
        summary = result['summary']
        plot_path = result['plot_path'] if result['plot_path'] else "No plot generated"
        return f"Task completed. Weather analysis around balloons completed.\n{summary}\nPlot generated at: {plot_path}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON input for BalloonWeatherAnalyzer"
    except Exception as e:
        return f"Error in BalloonWeatherAnalyzer: {str(e)}"

def balloon_plane_proximity_wrapper(input_str):
    try:
        input_dict = json.loads(input_str)
        result = BalloonPlaneProximity().analyze(**input_dict)
        summary = result['summary']
        plot_path = result['plot_path'] if result['plot_path'] else "No plot generated"
        return f"Task completed. Proximity check completed.\n{summary}\nPlot generated at: {plot_path}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON input for BalloonPlaneProximity"
    except Exception as e:
        return f"Error in BalloonPlaneProximity: {str(e)}"

def country_weather_monitor_wrapper(input_str):
    try:
        input_dict = json.loads(input_str)
        country_name = input_dict.get('country_name')
        if not country_name:
            return "Error: 'country_name' is required for CountryWeatherMonitor"
        
        result = CountryWeatherMonitor().monitor(**input_dict)
        summary = result['summary']
        plot_path = result['plot_path'] if result['plot_path'] else "No plot generated"
        return f"Task completed. Country weather monitoring completed.\n{summary}\nPlot generated at: {plot_path}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON input for CountryWeatherMonitor"
    except Exception as e:
        return f"Error in CountryWeatherMonitor: {str(e)}"

def balloon_coverage_analyzer_wrapper(input_str):
    try:
        input_dict = json.loads(input_str)
        result = BalloonCoverageAnalyzer().analyze(**input_dict)
        summary = result['summary']
        plot_path = result['plot_path'] if result['plot_path'] else "No plot generated"
        return f"Task completed. Coverage analysis completed.\n{summary}\nPlot generated at: {plot_path}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON input for BalloonCoverageAnalyzer"
    except Exception as e:
        return f"Error in BalloonCoverageAnalyzer: {str(e)}"

# Define Tools with Wrappers
tools = [
    Tool(
        name="BalloonTracker",
        func=balloon_tracker_wrapper,
        description="Tracks balloon positions. Input is a JSON string with 'num_balloons' (int, optional) and 'history_hours' (int, default 24). Example: '{\"num_balloons\": 5, \"history_hours\": 24}'"
    ),
    Tool(
        name="AltitudeAnalyzer",
        func=altitude_analyzer_wrapper,
        description="Analyzes balloon altitude patterns. Input is a JSON string with 'num_hours' (int, default 4). Example: '{\"num_hours\": 4}'"
    ),
    Tool(
        name="WindSpeedTool",
        func=wind_speed_tool_wrapper,
        description="Analyzes wind speeds. Input is a JSON string with 'hours' (int, default 2) and 'top_n' (int, default 5). Example: '{\"hours\": 2, \"top_n\": 5}'"
    ),
    Tool(
        name="BalloonWeatherAnalyzer",
        func=balloon_weather_analyzer_wrapper,
        description="Analyzes weather conditions around balloons. Input is a JSON string with 'n' (int, default 5) and 'plot' (bool, default true). Example: '{\"n\": 3, \"plot\": true}'"
    ),
    Tool(
        name="BalloonPlaneProximity",
        func=balloon_plane_proximity_wrapper,
        description="Checks proximity between balloons and aircraft. Input is a JSON string with 'distance_km' (float, default 100) and 'plot' (bool, default true). Example: '{\"distance_km\": 100, \"plot\": true}'"
    ),
    Tool(
        name="CountryWeatherMonitor",
        func=country_weather_monitor_wrapper,
        description="Monitors weather near a country's borders. Input is a JSON string with 'country_name' (str, required), 'buffer_km' (float, default 500), and 'plot' (bool, default true). Example: '{\"country_name\": \"United Kingdom\", \"buffer_km\": 300, \"plot\": true}'"
    ),
    Tool(
        name="BalloonCoverageAnalyzer",
        func=balloon_coverage_analyzer_wrapper,
        description="Analyzes balloon coverage and dead zones. Input is a JSON string with 'threshold_km' (float, default 500), 'hours' (int, default 24), and 'plot' (bool, default true). Example: '{\"threshold_km\": 500, \"hours\": 24, \"plot\": true}'"
    )
]

# Enhanced Prompt Template
template = """You are an advanced balloon monitoring and analysis system designed to assist users by analyzing balloon data, weather conditions, and related factors using specialized tools. Your goal is to interpret the user's question, select the appropriate tool, provide the correct input, and deliver a clear, actionable answer.

You have access to the following tools:

{tools}

Use this format to structure your response:
Question: [the input question you must answer]
Thought: [your reasoning about what to do]
Action: [the tool to use, one of {tool_names}]
Action Input: [a JSON-formatted string with the tool's input parameters]
Observation: [the result from the tool]

... (repeat Thought/Action/Action Input/Observation as needed)

Thought: [I now know the final answer]
Final Answer: [the concise answer to the original question]

**Instructions:**
- For each Action Input, provide a valid JSON string matching the tool's expected parameters (see tool descriptions for details).
- Use lowercase 'true' and 'false' for boolean values in JSON.
- If a parameter is optional and not specified by the user, use the default value or omit it if appropriate.
- Be thorough, conversational, and clear in your Thought steps to explain your reasoning.
- If the question is unclear, make reasonable assumptions and explain them.
- If the observation contains 'Task completed', summarize the results and provide the final answer, then stop.
- Some tools generate plots; include the plot path in your final answer if applicable and consider the task complete.
- If an error occurs (e.g., invalid country name), include the error message in the final answer and stop.
- Do not repeat tool calls unnecessarily after a successful result or plot generation.
- If you have reached a good result, you can end/stop, dont keep calling the function again and again
- If task is completed, terminate please!!!!!! please dont irritate me, if you get the result terminate immediately, sometimes the result is just a plot
- Dont try to replace one tool with another you will not get the right result


Begin!

Question: {input}
{agent_scratchpad}"""

# Custom Prompt Template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\n"
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

# Custom Output Parser
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(return_values={"output": llm_output.split("Final Answer:")[-1].strip()}, log=llm_output)
        regex = r"Action: (.*?)\nAction Input: (.*?)(\nObservation:|$)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)

# Initialize Agent Components
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
prompt = CustomPromptTemplate(template=template, tools=tools, input_variables=["input", "intermediate_steps"])
output_parser = CustomOutputParser()
llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = LLMSingleActionAgent(llm_chain=llm_chain, output_parser=output_parser, stop=["\nObservation:"], allowed_tools=[tool.name for tool in tools])
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# Example Usage
if __name__ == "__main__":
    # # Example 1: Check proximity to planes
    # question1 = "balloon locations on the map last 24 hours"
    # print("Running example 1...")
    # result1 = agent_executor.run(question1)
    # print("Result for example 1:", result1)

    # # Example 2: Check proximity to planes
    # question1 = "planes near balloons, 50 kms"
    # print("Running example 1...")
    # result1 = agent_executor.run(question1)
    # print("Result for example 1:", result1)

    # #Example 3: 
    # question2 = "calculate wind speed using the balloons, use last 2 hours"
    # print("\nRunning example 2...")
    # result2 = agent_executor.run(question2)
    # print("Result for example 2:", result2)

    # #Example 4: 
    # question2 = "top 5 balloons weather surroundings"
    # print("\nRunning example 2...")
    # result2 = agent_executor.run(question2)
    # print("Result for example 2:", result2)

    # #Example 5: 
    # question2 = "Altitude analysis"
    # print("\nRunning example 2...")
    # result2 = agent_executor.run(question2)
    # print("Result for example 2:", result2)

    # #Example 6: 
    # question2 = "check deadzones"
    # print("\nRunning example 2...")
    # result2 = agent_executor.run(question2)
    # print("Result for example 2:", result2)

    # #Example 7: 
    # question2 = "check how many balloons are surrounding France for weather analysis"
    # print("\nRunning example 2...")
    # result2 = agent_executor.run(question2)
    # print("Result for example 2:", result2)
# Function to Extract Plot Paths
def extract_plot_paths(text):
    """Extract one or more plot paths from the agent's response."""
    matches = re.findall(r"Plot generated at: (.+)", text)
    if not matches:
        matches = []
        # Handle AltitudeAnalyzer's multi-plot format
        plot_lines = re.findall(r"(\w+): (.+)", text)
        for key, path in plot_lines:
            if key in ["scatter", "distribution", "boxplot"] or path.endswith(('.png', '.jpg', '.jpeg')):
                matches.append(path)
    return matches if matches else None

# Function to extract plot paths from text
def extract_plot_paths(text):
    """Extracts file paths for generated plots from the agent's response."""
    # Matches "Plot generated at: path.png" or "scatter: path1.png"
    single_plots = re.findall(r"Plot generated at: (\S+)", text)
    multi_plots = re.findall(r"\b\w+:\s*(\S+\.png)", text)
    return single_plots + multi_plots

# Streamlit UI Setup

import streamlit as st
import os
import re
from langchain.agents import AgentExecutor
from contextlib import redirect_stdout
import io
st.title("🎈 Balloon Monitoring and Analysis System")
st.markdown("""
Welcome to the upgraded Balloon Monitoring and Analysis System! Ask anything about balloon positions, weather, or analysis. Check out the sample prompts below to get started.
""")

st.markdown("### Sample Prompts")
st.markdown("""
- **balloon locations on the map last 24 hours**
- **planes near balloons, 50 kms**
- **calculate wind speed using the balloons, use last 2 hours**
- **top 5 balloons weather surroundings**
- **Altitude analysis**
- **check deadzones**
- **check how many balloons are surrounding France for weather analysis**
""")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "plots" in message and message["plots"]:
            for plot_path in message["plots"]:
                if os.path.exists(plot_path):
                    st.image(plot_path, caption=f"Generated Plot: {os.path.basename(plot_path)}")
                else:
                    st.write(f"*Plot file not found: {plot_path}*")

# User input field
user_input = st.chat_input("Ask your question here...")

# Process user input
if user_input:
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Run the agent with verbose output capture
    try:
        with st.spinner("Processing your request..."):
            # Capture verbose output from the agent
            f = io.StringIO()
            with redirect_stdout(f):
                result = agent_executor.run(user_input)  # Assumes agent_executor is defined
            verbose_output = f.getvalue()
            final_answer = result
            plot_paths = extract_plot_paths(final_answer)
    except Exception as e:
        final_answer = f"Oops, something went wrong: {str(e)}"
        verbose_output = ""
        plot_paths = []
    
    # Display assistant response
    with st.chat_message("assistant"):
        # Show verbose output in a collapsible expander
        if verbose_output:
            with st.expander("Show Detailed Steps"):
                st.text(verbose_output)
        # Show the final answer
        st.markdown("**Final Answer**")
        st.markdown(final_answer)
        # Display any generated plots
        if plot_paths:
            for path in plot_paths:
                if os.path.exists(path):
                    st.image(path, caption=os.path.basename(path))
                else:
                    st.write(f"Plot file not found: {path}")
    
    # Add assistant message to history, including plots if any
    message = {"role": "assistant", "content": final_answer}
    if plot_paths:
        message["plots"] = plot_paths
    st.session_state.messages.append(message)
