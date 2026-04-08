"""
Route Optimization Module
Implements graph-based algorithms for finding optimal routes for autonomous EVs
Considers traffic, distance, and energy consumption
"""

import heapq
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np


class RouteObjective(Enum):
    """Optimization objectives"""
    FASTEST = "fastest"           # Minimize time
    SHORTEST = "shortest"         # Minimize distance
    ENERGY_EFFICIENT = "energy"   # Minimize energy consumption
    BALANCED = "balanced"         # Balance all factors


@dataclass
class Node:
    """Represents an intersection or waypoint in the road network"""
    id: int
    lat: float
    lon: float
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id


@dataclass
class Edge:
    """Represents a road segment between two nodes"""
    from_node: int
    to_node: int
    distance: float       # km
    speed_limit: float    # km/h
    elevation_gain: float # meters (positive = uphill, negative = downhill)
    road_type: str        # 'highway', 'primary', 'secondary', 'residential'
    
    def get_base_travel_time(self) -> float:
        """Calculate base travel time in hours"""
        return self.distance / self.speed_limit
    
    def get_energy_consumption(self, vehicle_efficiency: float = 0.2) -> float:
        """
        Calculate energy consumption in kWh
        
        Args:
            vehicle_efficiency: kWh per km (baseline)
        
        Returns:
            Energy consumption in kWh
        """
        # Base energy consumption
        base_energy = self.distance * vehicle_efficiency
        
        # Elevation penalty (uphill costs more energy)
        elevation_factor = 1.0
        if self.elevation_gain > 0:
            # Each 100m elevation gain adds 20% energy
            elevation_factor = 1.0 + (self.elevation_gain / 100) * 0.2
        elif self.elevation_gain < 0:
            # Downhill regenerates 10% energy back
            elevation_factor = 1.0 + (self.elevation_gain / 100) * 0.1
        
        return base_energy * elevation_factor


@dataclass
class Route:
    """Represents a complete route solution"""
    path: List[int]                    # List of node IDs
    total_distance: float              # km
    total_time: float                  # hours
    total_energy: float                # kWh
    segments: List[Tuple[int, int]]    # List of (from_node, to_node) pairs
    
    def get_cost_by_objective(self, objective: RouteObjective) -> float:
        """Get route cost based on optimization objective"""
        if objective == RouteObjective.FASTEST:
            return self.total_time
        elif objective == RouteObjective.SHORTEST:
            return self.total_distance
        elif objective == RouteObjective.ENERGY_EFFICIENT:
            return self.total_energy
        else:  # BALANCED
            # Normalize and combine all factors
            return self.total_time * 10 + self.total_distance + self.total_energy * 5


class RoadNetwork:
    """
    Represents the road network as a graph
    Supports efficient pathfinding operations
    """
    
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[Tuple[int, int], Edge] = {}
        self.adjacency: Dict[int, List[int]] = {}
        self.traffic_conditions: Dict[int, float] = {}  # segment_id -> speed_multiplier
    
    def add_node(self, node: Node):
        """Add a node to the network"""
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = []
    
    def add_edge(self, edge: Edge):
        """Add a directed edge to the network"""
        key = (edge.from_node, edge.to_node)
        self.edges[key] = edge
        
        if edge.from_node not in self.adjacency:
            self.adjacency[edge.from_node] = []
        self.adjacency[edge.from_node].append(edge.to_node)
    
    def get_neighbors(self, node_id: int) -> List[int]:
        """Get all neighboring nodes"""
        return self.adjacency.get(node_id, [])
    
    def get_edge(self, from_node: int, to_node: int) -> Optional[Edge]:
        """Get edge between two nodes"""
        return self.edges.get((from_node, to_node))
    
    def update_traffic(self, segment_speeds: Dict[int, float]):
        """
        Update traffic conditions
        segment_speeds: Dict mapping segment_id -> predicted_speed (km/h)
        """
        self.traffic_conditions = segment_speeds
    
    def get_actual_travel_time(self, from_node: int, to_node: int) -> float:
        """Get actual travel time considering current traffic"""
        edge = self.get_edge(from_node, to_node)
        if not edge:
            return float('inf')
        
        # Check if we have traffic data for this segment
        segment_id = hash((from_node, to_node)) % 100  # Simple segment ID
        
        if segment_id in self.traffic_conditions:
            actual_speed = self.traffic_conditions[segment_id]
            return edge.distance / actual_speed
        else:
            # Use base speed limit
            return edge.get_base_travel_time()
    
    def haversine_distance(self, node1_id: int, node2_id: int) -> float:
        """
        Calculate haversine distance between two nodes (for heuristic)
        Returns distance in km
        """
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        
        lat1, lon1 = math.radians(node1.lat), math.radians(node1.lon)
        lat2, lon2 = math.radians(node2.lat), math.radians(node2.lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return c * r


class RouteOptimizer:
    """
    Route optimization engine using various pathfinding algorithms
    """
    
    def __init__(self, network: RoadNetwork):
        self.network = network
        self.vehicle_efficiency = 0.2  # kWh per km (default for typical EV)
    
    def dijkstra(self, start_id: int, goal_id: int, 
                 objective: RouteObjective = RouteObjective.FASTEST) -> Optional[Route]:
        """
        Dijkstra's algorithm for shortest path
        Guarantees optimal solution but explores more nodes than A*
        
        Time Complexity: O((V + E) log V) with binary heap
        """
        
        # Priority queue: (cost, current_node, path)
        pq = [(0, start_id, [start_id])]
        visited = set()
        
        # Track best costs to each node
        best_costs = {start_id: 0}
        
        while pq:
            current_cost, current_node, path = heapq.heappop(pq)
            
            # Skip if already visited
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # Goal reached
            if current_node == goal_id:
                return self._construct_route(path, objective)
            
            # Explore neighbors
            for neighbor in self.network.get_neighbors(current_node):
                if neighbor in visited:
                    continue
                
                # Calculate cost based on objective
                edge_cost = self._calculate_edge_cost(current_node, neighbor, objective)
                new_cost = current_cost + edge_cost
                
                # Update if we found a better path
                if neighbor not in best_costs or new_cost < best_costs[neighbor]:
                    best_costs[neighbor] = new_cost
                    new_path = path + [neighbor]
                    heapq.heappush(pq, (new_cost, neighbor, new_path))
        
        return None  # No path found
    
    def a_star(self, start_id: int, goal_id: int,
               objective: RouteObjective = RouteObjective.FASTEST) -> Optional[Route]:
        """
        A* algorithm for optimal pathfinding
        Uses heuristic to guide search - faster than Dijkstra
        
        Time Complexity: O(E) in best case, O(V log V) in worst case
        """
        
        # Priority queue: (f_score, g_score, current_node, path)
        # f_score = g_score + h_score (actual cost + heuristic)
        pq = [(0, 0, start_id, [start_id])]
        visited = set()
        
        # Track best g_scores (actual costs from start)
        g_scores = {start_id: 0}
        
        while pq:
            f_score, g_score, current_node, path = heapq.heappop(pq)
            
            # Skip if already visited
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # Goal reached
            if current_node == goal_id:
                return self._construct_route(path, objective)
            
            # Explore neighbors
            for neighbor in self.network.get_neighbors(current_node):
                if neighbor in visited:
                    continue
                
                # Calculate actual cost (g_score)
                edge_cost = self._calculate_edge_cost(current_node, neighbor, objective)
                new_g_score = g_score + edge_cost
                
                # Update if we found a better path
                if neighbor not in g_scores or new_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = new_g_score
                    
                    # Calculate heuristic (h_score) - estimated cost to goal
                    h_score = self._heuristic(neighbor, goal_id, objective)
                    
                    # f_score = g_score + h_score
                    new_f_score = new_g_score + h_score
                    
                    new_path = path + [neighbor]
                    heapq.heappush(pq, (new_f_score, new_g_score, neighbor, new_path))
        
        return None  # No path found
    
    def find_optimal_route(self, start_id: int, goal_id: int,
                          objective: RouteObjective = RouteObjective.BALANCED,
                          algorithm: str = "a_star") -> Optional[Route]:
        """
        Main interface for route optimization
        
        Args:
            start_id: Starting node ID
            goal_id: Destination node ID
            objective: Optimization objective
            algorithm: "a_star" or "dijkstra"
        
        Returns:
            Optimal Route or None if no path exists
        """
        
        if algorithm == "a_star":
            return self.a_star(start_id, goal_id, objective)
        elif algorithm == "dijkstra":
            return self.dijkstra(start_id, goal_id, objective)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _calculate_edge_cost(self, from_node: int, to_node: int, 
                           objective: RouteObjective) -> float:
        """Calculate cost of traversing an edge based on objective"""
        
        edge = self.network.get_edge(from_node, to_node)
        if not edge:
            return float('inf')
        
        if objective == RouteObjective.FASTEST:
            # Cost = actual travel time considering traffic
            return self.network.get_actual_travel_time(from_node, to_node)
        
        elif objective == RouteObjective.SHORTEST:
            # Cost = distance
            return edge.distance
        
        elif objective == RouteObjective.ENERGY_EFFICIENT:
            # Cost = energy consumption
            return edge.get_energy_consumption(self.vehicle_efficiency)
        
        else:  # BALANCED
            # Weighted combination of all factors
            time_cost = self.network.get_actual_travel_time(from_node, to_node)
            distance_cost = edge.distance
            energy_cost = edge.get_energy_consumption(self.vehicle_efficiency)
            
            # Normalize and combine (weights can be tuned)
            return time_cost * 10 + distance_cost + energy_cost * 5
    
    def _heuristic(self, current_id: int, goal_id: int, 
                  objective: RouteObjective) -> float:
        """
        Heuristic function for A*
        Must be admissible (never overestimate) and consistent
        """
        
        # Straight-line distance as base heuristic
        h_distance = self.network.haversine_distance(current_id, goal_id)
        
        if objective == RouteObjective.SHORTEST:
            return h_distance
        
        elif objective == RouteObjective.FASTEST:
            # Assume max speed of 100 km/h for heuristic
            max_speed = 100.0
            return h_distance / max_speed
        
        elif objective == RouteObjective.ENERGY_EFFICIENT:
            # Assume best-case energy efficiency
            best_efficiency = self.vehicle_efficiency * 0.8
            return h_distance * best_efficiency
        
        else:  # BALANCED
            # Use fastest time heuristic as it's most conservative
            return h_distance / 100.0
    
    def _construct_route(self, path: List[int], objective: RouteObjective) -> Route:
        """Construct a Route object from a path"""
        
        total_distance = 0.0
        total_time = 0.0
        total_energy = 0.0
        segments = []
        
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            
            edge = self.network.get_edge(from_node, to_node)
            if edge:
                total_distance += edge.distance
                total_time += self.network.get_actual_travel_time(from_node, to_node)
                total_energy += edge.get_energy_consumption(self.vehicle_efficiency)
                segments.append((from_node, to_node))
        
        return Route(
            path=path,
            total_distance=total_distance,
            total_time=total_time,
            total_energy=total_energy,
            segments=segments
        )


def create_mock_city_network() -> RoadNetwork:
    """
    Create a mock city road network for testing
    Simulates a 10x10 grid of intersections
    """
    
    network = RoadNetwork()
    
    # Create 10x10 grid of nodes
    grid_size = 10
    node_id = 0
    
    # Base coordinates (simulating a city)
    base_lat = 28.6139  # Delhi, India
    base_lon = 77.2090
    
    # Create nodes
    for i in range(grid_size):
        for j in range(grid_size):
            lat = base_lat + (i * 0.01)  # ~1 km per 0.01 degrees
            lon = base_lon + (j * 0.01)
            
            node = Node(id=node_id, lat=lat, lon=lon)
            network.add_node(node)
            node_id += 1
    
    # Create edges (roads connecting nodes)
    for i in range(grid_size):
        for j in range(grid_size):
            current_node = i * grid_size + j
            
            # Connect to right neighbor
            if j < grid_size - 1:
                right_node = i * grid_size + (j + 1)
                
                # Vary road types
                if i == 0 or i == grid_size - 1:
                    road_type = "highway"
                    speed_limit = 80.0
                elif i == grid_size // 2:
                    road_type = "primary"
                    speed_limit = 60.0
                else:
                    road_type = "secondary"
                    speed_limit = 40.0
                
                # Random elevation changes
                elevation_gain = np.random.uniform(-10, 10)
                
                edge = Edge(
                    from_node=current_node,
                    to_node=right_node,
                    distance=1.0,  # 1 km
                    speed_limit=speed_limit,
                    elevation_gain=elevation_gain,
                    road_type=road_type
                )
                network.add_edge(edge)
                
                # Add reverse direction
                edge_reverse = Edge(
                    from_node=right_node,
                    to_node=current_node,
                    distance=1.0,
                    speed_limit=speed_limit,
                    elevation_gain=-elevation_gain,
                    road_type=road_type
                )
                network.add_edge(edge_reverse)
            
            # Connect to bottom neighbor
            if i < grid_size - 1:
                bottom_node = (i + 1) * grid_size + j
                
                # Vary road types
                if j == 0 or j == grid_size - 1:
                    road_type = "highway"
                    speed_limit = 80.0
                elif j == grid_size // 2:
                    road_type = "primary"
                    speed_limit = 60.0
                else:
                    road_type = "secondary"
                    speed_limit = 40.0
                
                elevation_gain = np.random.uniform(-10, 10)
                
                edge = Edge(
                    from_node=current_node,
                    to_node=bottom_node,
                    distance=1.0,
                    speed_limit=speed_limit,
                    elevation_gain=elevation_gain,
                    road_type=road_type
                )
                network.add_edge(edge)
                
                # Add reverse direction
                edge_reverse = Edge(
                    from_node=bottom_node,
                    to_node=current_node,
                    distance=1.0,
                    speed_limit=speed_limit,
                    elevation_gain=-elevation_gain,
                    road_type=road_type
                )
                network.add_edge(edge_reverse)
    
    return network


def test_route_optimization():
    """Test the route optimization algorithms"""
    
    print("=" * 60)
    print("TESTING ROUTE OPTIMIZATION ALGORITHMS")
    print("=" * 60)
    
    # Create mock network
    print("\nCreating mock city network (10x10 grid)...")
    network = create_mock_city_network()
    print(f"  Nodes: {len(network.nodes)}")
    print(f"  Edges: {len(network.edges)}")
    
    # Create optimizer
    optimizer = RouteOptimizer(network)
    
    # Test routes
    start_node = 0     # Top-left corner
    goal_node = 99     # Bottom-right corner
    
    print(f"\nFinding routes from node {start_node} to node {goal_node}")
    print("-" * 60)
    
    # Test different objectives
    objectives = [
        (RouteObjective.FASTEST, "Fastest Route"),
        (RouteObjective.SHORTEST, "Shortest Route"),
        (RouteObjective.ENERGY_EFFICIENT, "Most Energy-Efficient Route"),
        (RouteObjective.BALANCED, "Balanced Route")
    ]
    
    for objective, name in objectives:
        print(f"\n{name} (A* Algorithm):")
        route = optimizer.a_star(start_node, goal_node, objective)
        
        if route:
            print(f"  Path: {route.path[:5]}...{route.path[-5:]}")  # Show first and last 5 nodes
            print(f"  Total nodes: {len(route.path)}")
            print(f"  Distance: {route.total_distance:.2f} km")
            print(f"  Time: {route.total_time * 60:.1f} minutes")
            print(f"  Energy: {route.total_energy:.2f} kWh")
        else:
            print("  No route found!")
    
    # Compare A* vs Dijkstra
    print("\n" + "=" * 60)
    print("COMPARING A* vs DIJKSTRA")
    print("=" * 60)
    
    import time
    
    # A* timing
    start_time = time.time()
    route_astar = optimizer.a_star(start_node, goal_node, RouteObjective.FASTEST)
    astar_time = time.time() - start_time
    
    # Dijkstra timing
    start_time = time.time()
    route_dijkstra = optimizer.dijkstra(start_node, goal_node, RouteObjective.FASTEST)
    dijkstra_time = time.time() - start_time
    
    print(f"\nA* Algorithm:")
    print(f"  Computation time: {astar_time*1000:.2f} ms")
    print(f"  Route distance: {route_astar.total_distance:.2f} km" if route_astar else "  No route found")
    
    print(f"\nDijkstra's Algorithm:")
    print(f"  Computation time: {dijkstra_time*1000:.2f} ms")
    print(f"  Route distance: {route_dijkstra.total_distance:.2f} km" if route_dijkstra else "  No route found")
    
    if route_astar and route_dijkstra:
        print(f"\nSpeedup: A* is {dijkstra_time/astar_time:.2f}x faster")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_route_optimization()