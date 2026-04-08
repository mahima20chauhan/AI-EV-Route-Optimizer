"""
OpenStreetMap Integration
Load real road network data from OpenStreetMap
"""

import requests
import math
from typing import Dict, Optional
from route_optimizer import RoadNetwork, Node, Edge


class OSMLoader:

    def __init__(self):
        self.overpass_url = "https://overpass-api.de/api/interpreter"

    # ✅ DEFAULT RADIUS = 30 KM
    def load_network_for_area(self, lat: float, lon: float, radius_km: float = 30.0) -> RoadNetwork:

        print(f"Loading OSM data for area: ({lat}, {lon}), radius={radius_km}km")

        radius_m = radius_km * 1000

        # ✅ Increased timeout for big area
        query = f"""
        [out:json][timeout:60];
        (
          way["highway"]["highway"!~"footway|path|steps|cycleway"]
          (around:{radius_m},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
        """

        try:
            response = requests.post(
                self.overpass_url,
                data={'data': query},
                timeout=60
            )

            if response.status_code != 200:
                print("⚠ API failed → using fallback")
                return self._create_fallback_network(lat, lon, radius_km)

            data = response.json()
            network = self._parse_osm_data(data)

            print(f"✓ Loaded {len(network.nodes)} nodes and {len(network.edges)} edges")
            return network

        except Exception as e:
            print(f"⚠ Error: {e}")
            return self._create_fallback_network(lat, lon, radius_km)

    def _parse_osm_data(self, data: Dict) -> RoadNetwork:

        network = RoadNetwork()
        osm_nodes = {}
        osm_ways = []

        for element in data['elements']:
            if element['type'] == 'node':
                osm_nodes[element['id']] = Node(
                    id=element['id'],
                    lat=element['lat'],
                    lon=element['lon']
                )

            elif element['type'] == 'way':
                osm_ways.append(element)

        for node in osm_nodes.values():
            network.add_node(node)

        for way in osm_ways:
            nodes = way.get('nodes', [])
            tags = way.get('tags', {})

            speed = self._get_speed(tags)

            for i in range(len(nodes) - 1):
                if nodes[i] not in osm_nodes or nodes[i+1] not in osm_nodes:
                    continue

                n1 = osm_nodes[nodes[i]]
                n2 = osm_nodes[nodes[i+1]]

                dist = self._haversine(n1.lat, n1.lon, n2.lat, n2.lon)

                edge = Edge(nodes[i], nodes[i+1], dist, speed, 0, tags.get('highway'))
                network.add_edge(edge)

                if tags.get('oneway') != 'yes':
                    edge2 = Edge(nodes[i+1], nodes[i], dist, speed, 0, tags.get('highway'))
                    network.add_edge(edge2)

        return network

    def _get_speed(self, tags):
        if 'maxspeed' in tags:
            try:
                return float(tags['maxspeed'].split()[0])
            except:
                pass

        return {
            'motorway': 100,
            'primary': 70,
            'secondary': 60,
            'residential': 30
        }.get(tags.get('highway'), 40)

    def _haversine(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return 6371 * 2 * math.asin(math.sqrt(a))

    def _create_fallback_network(self, lat, lon, radius_km):
        from route_optimizer import create_mock_city_network
        print("⚠ Using fallback network")
        return create_mock_city_network()

    def find_nearest_node(self, network: RoadNetwork, lat: float, lon: float) -> Optional[int]:
        min_dist = float('inf')
        best = None

        for nid, node in network.nodes.items():
            d = self._haversine(lat, lon, node.lat, node.lon)
            if d < min_dist:
                min_dist = d
                best = nid

        return best