from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys, os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'route_optimizer'))

from route_optimizer import (
    RouteOptimizer,
    create_mock_city_network
)
from osm_loader import OSMLoader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

network = None
optimizer = None
osm_loader = OSMLoader()

# -------- MODELS --------

class Coordinate(BaseModel):
    lat: float
    lon: float

class RouteRequest(BaseModel):
    start: Coordinate
    destination: Coordinate

# -------- STARTUP --------

@app.on_event("startup")
async def startup():
    global network, optimizer

    try:
        network = osm_loader.load_network_for_area(28.6139, 77.2090, radius_km=30.0)
    except:
        network = create_mock_city_network()

    optimizer = RouteOptimizer(network)

# -------- ROOT --------

@app.get("/")
def root():
    return {"message": "API running 🚀"}

# -------- MAIN FIXED ROUTE API --------

@app.post("/api/optimize")
def optimize(request: RouteRequest):
    try:
        start_node = osm_loader.find_nearest_node(network, request.start.lat, request.start.lon)
        end_node = osm_loader.find_nearest_node(network, request.destination.lat, request.destination.lon)

        route = optimizer.find_optimal_route(start_node, end_node)

        if not route:
            raise HTTPException(status_code=404, detail="No route found")

        # ✅ CLEAN PATH (remove loops)
        clean_path = []
        visited = set()

        for node in route.path:
            if node not in visited:
                clean_path.append(node)
                visited.add(node)

        # ✅ CONVERT TO LAT LON WAYPOINTS
        waypoints = []
        for node_id in clean_path:
            node = network.nodes[node_id]
            waypoints.append({
                "lat": node.lat,
                "lon": node.lon
            })

        return {
            "total_distance": route.total_distance,
            "total_time": route.total_time,
            "total_energy": route.total_energy,
            "waypoints": waypoints
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))