import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import './App.css';

// Fix marker icons
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});

L.Marker.prototype.options.icon = DefaultIcon;

const API_BASE_URL = 'http://localhost:8000';

function App() {

  const [startCoords, setStartCoords] = useState({ lat: 28.6139, lon: 77.2090 });
  const [destCoords, setDestCoords] = useState({ lat: 28.4595, lon: 77.0266 });
  const [routeData, setRouteData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const optimizeRoute = async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start: startCoords,
          destination: destCoords
        })
      });

      if (!res.ok) throw new Error(res.status);

      const data = await res.json();

      // ✅ DIRECTLY USE BACKEND WAYPOINTS (NO RANDOM FAKE PATH)
      const cleanPath = data.waypoints.map(wp => [wp.lat, wp.lon]);

      setRouteData({
        ...data,
        routePath: cleanPath
      });

    } catch (err) {
      console.error(err);
      setError("⚠ Failed to fetch route");
    }

    setLoading(false);
  };

  const routePath = routeData ? routeData.routePath : [];

  return (
    <div className="App">

      <div className="header">
        <h1>🚗 EV Route Optimizer</h1>
        <p>Smart navigation with AI + traffic prediction</p>
      </div>

      <div className="main-container">

        {/* LEFT PANEL */}
        <div className="control-panel">

          <h2>📍 Route Setup</h2>

          <label>Start Location</label>
          <div className="coord-inputs">
            <input
              type="number"
              value={startCoords.lat}
              onChange={e => setStartCoords({ ...startCoords, lat: +e.target.value })}
            />
            <input
              type="number"
              value={startCoords.lon}
              onChange={e => setStartCoords({ ...startCoords, lon: +e.target.value })}
            />
          </div>

          <label>Destination</label>
          <div className="coord-inputs">
            <input
              type="number"
              value={destCoords.lat}
              onChange={e => setDestCoords({ ...destCoords, lat: +e.target.value })}
            />
            <input
              type="number"
              value={destCoords.lon}
              onChange={e => setDestCoords({ ...destCoords, lon: +e.target.value })}
            />
          </div>

          <button onClick={optimizeRoute} disabled={loading}>
            {loading ? "Optimizing..." : "🚀 Optimize Route"}
          </button>

          {error && <div className="error">{error}</div>}

          {routeData && (
            <div className="mini-stats">
              <p>📏 {routeData.total_distance?.toFixed(2)} km</p>
              <p>⏱ {(routeData.total_time * 60)?.toFixed(1)} min</p>
              <p>⚡ {routeData.total_energy?.toFixed(2)} kWh</p>
            </div>
          )}

        </div>

        {/* MAP */}
        <div className="map-panel">
          <MapContainer
            center={[startCoords.lat, startCoords.lon]}
            zoom={12}
            className="map"
          >
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

            <Marker position={[startCoords.lat, startCoords.lon]}>
              <Popup>Start</Popup>
            </Marker>

            <Marker position={[destCoords.lat, destCoords.lon]}>
              <Popup>End</Popup>
            </Marker>

            {routePath.length > 0 && (
              <Polyline positions={routePath} color="#4f46e5" />
            )}

            {routePath.length > 0 && <FitBounds path={routePath} />}
          </MapContainer>
        </div>

      </div>
    </div>
  );
}

function FitBounds({ path }) {
  const map = useMap();

  useEffect(() => {
    if (path.length > 0) {
      map.fitBounds(L.latLngBounds(path), { padding: [40, 40] });
    }
  }, [path]);

  return null;
}

export default App;