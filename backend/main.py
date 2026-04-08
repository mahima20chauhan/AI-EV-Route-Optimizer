"""
FastAPI Backend for Intelligent Route Optimization
Autonomous Electric Vehicle Route Planning System
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_model'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'route_optimizer'))

from traffic_predictor import TrafficPredictor
from route_optimizer import (
    RoadNetwork, RouteOptimizer, RouteObjective,
    create_mock_city_network
)
from osm_loader import OSMLoader

# Initialize FastAPI app
app = FastAPI(
    title="AEV Route Optimization API",
    description="Intelligent route optimization for Autonomous Electric Vehicles",
    version="1.0.0"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
traffic_predictor: Optional[TrafficPredictor] = None
network: Optional[RoadNetwork] = None
optimizer: Optional[RouteOptimizer] = None
osm_loader: OSMLoader = OSMLoader()


# ==================== Pydantic Models ====================

class Coordinate(BaseModel):
    """Geographic coordinate"""
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")


class VehicleSpecs(BaseModel):
    """Electric vehicle specifications"""
    battery_capacity: float = Field(60.0, description="Battery capacity in kWh")
    efficiency: float = Field(0.2, description="Energy consumption in kWh/km")
    current_charge: float = Field(100.0, description="Current battery charge percentage")


class RouteRequest(BaseModel):
    """Request for route optimization"""
    start: Coordinate
    destination: Coordinate
    vehicle: VehicleSpecs = VehicleSpecs()
    objective: str = Field("balanced", description="fastest, shortest, energy, or balanced")
    algorithm: str = Field("a_star", description="a_star or dijkstra")
    consider_traffic: bool = Field(True, description="Include traffic predictions")
    weather: int = Field(0, description="Weather condition: 0=clear, 1=rain, 2=snow")


class WaypointInfo(BaseModel):
    """Information about a waypoint on the route"""
    node_id: int
    lat: float
    lon: float
    distance_from_start: float
    time_from_start: float
    energy_from_start: float


class RouteResponse(BaseModel):
    """Response with optimized route"""
    route_id: str
    waypoints: List[WaypointInfo]
    total_distance: float
    total_time: float  # in hours
    total_energy: float
    eta: str  # Estimated time of arrival
    battery_remaining: float  # Percentage
    algorithm_used: str
    objective: str
    computation_time: float  # in milliseconds


class TrafficPredictionRequest(BaseModel):
    """Request for traffic prediction"""
    road_segments: List[int]
    weather: int = 0


class TrafficPredictionResponse(BaseModel):
    """Traffic prediction for road segments"""
    predictions: Dict[int, float]  # segment_id -> predicted_speed
    timestamp: str


class NetworkStatsResponse(BaseModel):
    """Network statistics"""
    num_nodes: int
    num_edges: int
    network_type: str  # "osm" or "mock"
    area_description: str


# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global traffic_predictor, network, optimizer
    
    print("=" * 60)
    print("INITIALIZING AEV ROUTE OPTIMIZATION SYSTEM")
    print("=" * 60)
    
    # Initialize traffic predictor
    try:
        print("\n1. Loading traffic prediction model...")
        traffic_predictor = TrafficPredictor(
            model_path='../ml_model/traffic_model_production.pth',
            scaler_path='../ml_model/traffic_scaler.pkl',
            metadata_path='../ml_model/model_metadata.json'
        )
        print("✓ Traffic predictor loaded")
    except FileNotFoundError:
        print("⚠ Traffic model not found. Run train_traffic_model.py first.")
        print("  Using mock predictions for now.")
        traffic_predictor = None
    
    # Initialize road network
    print("\n2. Loading road network...")
    try:
        # Try to load real OSM data for a default location (Delhi)
        network = osm_loader.load_network_for_area(28.6139, 77.2090, radius_km=30.0)
        print("✓ Road network loaded")
    except Exception as e:
        print(f"⚠ Could not load OSM data: {e}")
        print("  Using mock network instead.")
        network = create_mock_city_network()
        print("✓ Mock network created")
    
    # Initialize route optimizer
    print("\n3. Initializing route optimizer...")
    optimizer = RouteOptimizer(network)
    print("✓ Route optimizer ready")
    
    print("\n" + "=" * 60)
    print("SYSTEM READY - API Server Started")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\nShutting down AEV Route Optimization System...")


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "AEV Route Optimization API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "optimize": "/api/optimize - POST - Get optimized route",
            "traffic": "/api/traffic/predict - POST - Get traffic predictions",
            "network": "/api/network/stats - GET - Get network statistics",
            "health": "/health - GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "traffic_predictor": traffic_predictor is not None,
            "network": network is not None,
            "optimizer": optimizer is not None
        }
    }


@app.get("/api/network/stats", response_model=NetworkStatsResponse)
async def get_network_stats():
    """Get statistics about the road network"""
    
    if not network:
        raise HTTPException(status_code=503, detail="Network not initialized")
    
    return NetworkStatsResponse(
        num_nodes=len(network.nodes),
        num_edges=len(network.edges),
        network_type="osm" if len(network.nodes) > 100 else "mock",
        area_description=f"Road network with {len(network.nodes)} intersections"
    )


@app.post("/api/traffic/predict", response_model=TrafficPredictionResponse)
async def predict_traffic(request: TrafficPredictionRequest):
    """Predict traffic for specified road segments"""
    
    if not traffic_predictor:
        # Return mock predictions
        predictions = {
            seg: 50.0 + (seg % 30 - 15)  # Mock varying speeds
            for seg in request.road_segments
        }
    else:
        current_time = datetime.now()
        predictions = traffic_predictor.predict_route_traffic(
            request.road_segments,
            current_time,
            request.weather
        )
    
    return TrafficPredictionResponse(
        predictions=predictions,
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/optimize", response_model=RouteResponse)
async def optimize_route(request: RouteRequest):
    """
    Find optimal route for autonomous EV
    Main endpoint for route optimization
    """
    
    if not network or not optimizer:
        raise HTTPException(status_code=503, detail="Optimization service not available")
    
    import time
    start_time = time.time()
    
    # Find nearest nodes to start and destination
    start_node = osm_loader.find_nearest_node(network, request.start.lat, request.start.lon)
    dest_node = osm_loader.find_nearest_node(network, request.destination.lat, request.destination.lon)
    
    if not start_node or not dest_node:
        raise HTTPException(status_code=404, detail="Could not find nodes near specified coordinates")
    
    # Convert objective string to enum
    objective_map = {
        "fastest": RouteObjective.FASTEST,
        "shortest": RouteObjective.SHORTEST,
        "energy": RouteObjective.ENERGY_EFFICIENT,
        "balanced": RouteObjective.BALANCED
    }
    objective = objective_map.get(request.objective, RouteObjective.BALANCED)
    
    # Update traffic conditions if requested
    if request.consider_traffic and traffic_predictor:
        # Get all route segments that might be used
        all_segments = list(range(100))  # Simplified
        predictions = traffic_predictor.predict_route_traffic(
            all_segments,
            datetime.now(),
            request.weather
        )
        network.update_traffic(predictions)
    
    # Set vehicle efficiency
    optimizer.vehicle_efficiency = request.vehicle.efficiency
    
    # Find optimal route
    route = optimizer.find_optimal_route(
        start_node,
        dest_node,
        objective,
        request.algorithm
    )
    
    if not route:
        raise HTTPException(status_code=404, detail="No route found")
    
    # Calculate computation time
    computation_time = (time.time() - start_time) * 1000  # ms
    
    # Build waypoints
    waypoints = []
    cumulative_distance = 0.0
    cumulative_time = 0.0
    cumulative_energy = 0.0
    
    for i, node_id in enumerate(route.path):
        node = network.nodes[node_id]
        
        waypoints.append(WaypointInfo(
            node_id=node_id,
            lat=node.lat,
            lon=node.lon,
            distance_from_start=cumulative_distance,
            time_from_start=cumulative_time,
            energy_from_start=cumulative_energy
        ))
        
        # Update cumulative values for next waypoint
        if i < len(route.path) - 1:
            next_node_id = route.path[i + 1]
            edge = network.get_edge(node_id, next_node_id)
            if edge:
                cumulative_distance += edge.distance
                cumulative_time += network.get_actual_travel_time(node_id, next_node_id)
                cumulative_energy += edge.get_energy_consumption(request.vehicle.efficiency)
    
    # Calculate ETA
    eta_time = datetime.now() + pd.Timedelta(hours=route.total_time)
    eta_str = eta_time.strftime("%I:%M %p")
    
    # Calculate battery remaining
    energy_used_pct = (route.total_energy / request.vehicle.battery_capacity) * 100
    battery_remaining = request.vehicle.current_charge - energy_used_pct
    
    # Generate route ID
    import hashlib
    route_id = hashlib.md5(
        f"{start_node}{dest_node}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]
    
    return RouteResponse(
        route_id=route_id,
        waypoints=waypoints,
        total_distance=route.total_distance,
        total_time=route.total_time,
        total_energy=route.total_energy,
        eta=eta_str,
        battery_remaining=max(0, battery_remaining),
        algorithm_used=request.algorithm,
        objective=request.objective,
        computation_time=computation_time
    )


# Import pandas for timedelta
import pandas as pd


# ==================== WebSocket for Real-time Updates ====================

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


@app.websocket("/ws/traffic")
async def websocket_traffic(websocket: WebSocket):
    """
    WebSocket endpoint for real-time traffic updates
    Sends traffic predictions every 5 seconds
    """
    
    await manager.connect(websocket)
    
    try:
        while True:
            # Wait for client message (can send route info)
            data = await websocket.receive_json()
            
            # Predict traffic for specified segments
            segments = data.get('segments', list(range(20)))
            weather = data.get('weather', 0)
            
            if traffic_predictor:
                predictions = traffic_predictor.predict_route_traffic(
                    segments,
                    datetime.now(),
                    weather
                )
            else:
                predictions = {seg: 50.0 for seg in segments}
            
            # Send traffic update
            await websocket.send_json({
                "type": "traffic_update",
                "predictions": predictions,
                "timestamp": datetime.now().isoformat()
            })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ==================== Additional Utility Endpoints ====================

@app.get("/api/algorithms")
async def get_algorithms():
    """Get available routing algorithms and objectives"""
    return {
        "algorithms": ["a_star", "dijkstra"],
        "objectives": ["fastest", "shortest", "energy", "balanced"],
        "descriptions": {
            "a_star": "A* algorithm - Optimal and fast using heuristics",
            "dijkstra": "Dijkstra's algorithm - Guaranteed optimal, explores more nodes",
            "fastest": "Minimize travel time",
            "shortest": "Minimize distance",
            "energy": "Minimize energy consumption",
            "balanced": "Balance time, distance, and energy"
        }
    }


@app.get("/api/weather")
async def get_weather_info():
    """Get weather condition codes"""
    return {
        "codes": {
            "0": "Clear",
            "1": "Rain",
            "2": "Snow"
        },
        "note": "Weather affects traffic predictions and route optimization"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("STARTING AEV ROUTE OPTIMIZATION API SERVER")
    print("=" * 60)
    print("\nAPI will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")