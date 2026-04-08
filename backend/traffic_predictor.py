"""
Traffic Prediction Inference
Real-time traffic speed prediction for route optimization
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
import json
from datetime import datetime
from typing import List, Dict, Tuple


class TrafficLSTM(nn.Module):
    """LSTM model architecture (must match training)"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(TrafficLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out


class TrafficPredictor:
    """
    Traffic prediction service for production use
    Loads trained model and provides fast inference
    """
    
    def __init__(self, model_path='traffic_model_production.pth', 
                 scaler_path='traffic_scaler.pkl',
                 metadata_path='model_metadata.json'):
        """Initialize predictor with trained model"""
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = TrafficLSTM(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.seq_length = checkpoint['seq_length']
        
        print(f"✓ Traffic predictor loaded successfully")
        print(f"  Device: {self.device}")
        print(f"  Sequence length: {self.seq_length}")
    
    def predict_speed(self, historical_data: List[Dict], 
                     current_time: datetime = None,
                     road_segment: int = 0,
                     weather: int = 0) -> float:
        """
        Predict traffic speed for the next time step
        
        Args:
            historical_data: List of dicts with 'hour', 'day_of_week', 'weather', 
                           'road_segment', 'speed' for past time steps
            current_time: Current datetime (defaults to now)
            road_segment: Road segment ID (0-99)
            weather: Weather condition (0=clear, 1=rain, 2=snow)
        
        Returns:
            Predicted speed in km/h
        """
        
        if current_time is None:
            current_time = datetime.now()
        
        # If we have enough historical data, use it
        if len(historical_data) >= self.seq_length:
            sequence_data = historical_data[-self.seq_length:]
        else:
            # Generate mock historical data if not enough
            sequence_data = self._generate_mock_sequence(
                current_time, road_segment, weather
            )
        
        # Prepare features
        features = []
        for data_point in sequence_data:
            features.append([
                data_point['hour'],
                data_point['day_of_week'],
                data_point['weather'],
                data_point['road_segment'],
                data_point['speed']
            ])
        
        features = np.array(features)
        
        # Normalize
        features_normalized = self.scaler.transform(features)
        
        # Reshape for model: (1, seq_length, num_features)
        X = torch.FloatTensor(features_normalized).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(X)
        
        # Denormalize prediction
        # The prediction is normalized speed (last feature)
        dummy_features = np.zeros((1, 5))
        dummy_features[0, -1] = prediction.item()
        denormalized = self.scaler.inverse_transform(dummy_features)
        predicted_speed = denormalized[0, -1]
        
        # Clip to realistic range
        predicted_speed = max(10.0, min(120.0, predicted_speed))
        
        return predicted_speed
    
    def predict_route_traffic(self, route_segments: List[int],
                            current_time: datetime = None,
                            weather: int = 0) -> Dict[int, float]:
        """
        Predict traffic for multiple road segments along a route
        
        Args:
            route_segments: List of road segment IDs
            current_time: Current datetime
            weather: Weather condition
        
        Returns:
            Dict mapping segment_id -> predicted_speed
        """
        
        if current_time is None:
            current_time = datetime.now()
        
        predictions = {}
        
        for segment in route_segments:
            # Generate mock historical data for this segment
            historical = self._generate_mock_sequence(
                current_time, segment, weather
            )
            
            # Predict speed
            speed = self.predict_speed(
                historical, current_time, segment, weather
            )
            
            predictions[segment] = speed
        
        return predictions
    
    def _generate_mock_sequence(self, current_time: datetime,
                                road_segment: int, weather: int) -> List[Dict]:
        """
        Generate realistic mock historical sequence
        Used when actual historical data is not available
        """
        
        sequence = []
        
        for i in range(self.seq_length):
            # Go back in time
            time_offset = self.seq_length - i
            timestamp = current_time - pd.Timedelta(minutes=5 * time_offset)
            
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Base speed with time-of-day effects
            base_speed = 50
            
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                time_factor = 0.6
            elif 22 <= hour or hour <= 5:
                time_factor = 1.2
            else:
                time_factor = 1.0
            
            weekend_factor = 1.1 if day_of_week >= 5 else 1.0
            weather_factor = 1.0 if weather == 0 else (0.8 if weather == 1 else 0.6)
            
            speed = base_speed * time_factor * weekend_factor * weather_factor
            speed += np.random.normal(0, 3)
            speed = max(10, min(120, speed))
            
            sequence.append({
                'hour': hour,
                'day_of_week': day_of_week,
                'weather': weather,
                'road_segment': road_segment,
                'speed': speed
            })
        
        return sequence
    
    def get_traffic_category(self, speed: float) -> str:
        """Convert speed to traffic condition category"""
        
        if speed >= 60:
            return "free_flow"
        elif speed >= 40:
            return "moderate"
        elif speed >= 25:
            return "congested"
        else:
            return "heavy_congestion"
    
    def get_traffic_multiplier(self, speed: float) -> float:
        """
        Get time multiplier for route optimization
        Lower speed = higher multiplier (takes longer)
        """
        
        # Normal speed is 50 km/h
        # Multiplier = normal_speed / actual_speed
        normal_speed = 50.0
        return normal_speed / max(speed, 10)  # Avoid division by zero


# Import pandas for timedelta (used in mock sequence generation)
import pandas as pd


def test_predictor():
    """Test the traffic predictor"""
    
    print("=" * 60)
    print("TESTING TRAFFIC PREDICTOR")
    print("=" * 60)
    
    # Initialize predictor
    predictor = TrafficPredictor(
        model_path='traffic_model_production.pth',
        scaler_path='traffic_scaler.pkl',
        metadata_path='model_metadata.json'
    )
    
    # Test 1: Single prediction
    print("\nTest 1: Predict speed for morning rush hour")
    test_time = datetime(2024, 1, 15, 8, 30)  # Monday, 8:30 AM
    historical = predictor._generate_mock_sequence(test_time, road_segment=42, weather=0)
    speed = predictor.predict_speed(historical, test_time, 42, 0)
    category = predictor.get_traffic_category(speed)
    
    print(f"  Time: {test_time.strftime('%A, %I:%M %p')}")
    print(f"  Road segment: 42")
    print(f"  Weather: Clear")
    print(f"  Predicted speed: {speed:.1f} km/h")
    print(f"  Traffic condition: {category}")
    print(f"  Time multiplier: {predictor.get_traffic_multiplier(speed):.2f}x")
    
    # Test 2: Route prediction
    print("\nTest 2: Predict traffic for route segments")
    route_segments = [10, 25, 42, 67, 89]
    predictions = predictor.predict_route_traffic(route_segments, test_time, weather=0)
    
    print(f"  Route has {len(route_segments)} segments:")
    for segment, pred_speed in predictions.items():
        condition = predictor.get_traffic_category(pred_speed)
        print(f"    Segment {segment}: {pred_speed:.1f} km/h ({condition})")
    
    # Test 3: Different times of day
    print("\nTest 3: Speed predictions throughout the day")
    test_hours = [6, 8, 12, 17, 22]
    for hour in test_hours:
        test_time = datetime(2024, 1, 15, hour, 0)
        historical = predictor._generate_mock_sequence(test_time, 42, 0)
        speed = predictor.predict_speed(historical, test_time, 42, 0)
        print(f"  {hour:02d}:00 - {speed:.1f} km/h ({predictor.get_traffic_category(speed)})")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests if models exist, otherwise provide instructions
    try:
        test_predictor()
    except FileNotFoundError as e:
        print(f"\n⚠ Model files not found: {e}")
        print("\nPlease run 'train_traffic_model.py' first to train the model.")
        print("This will generate:")
        print("  - traffic_model_production.pth")
        print("  - traffic_scaler.pkl")
        print("  - model_metadata.json")