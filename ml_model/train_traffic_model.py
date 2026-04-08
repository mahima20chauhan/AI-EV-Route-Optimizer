"""
Traffic Prediction Model - LSTM Implementation
Trains a deep learning model to predict traffic conditions for route optimization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class TrafficDataset(Dataset):
    """Custom PyTorch Dataset for traffic data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TrafficLSTM(nn.Module):
    """
    LSTM Neural Network for Traffic Speed Prediction
    
    Architecture:
    - Input layer: Multiple features (time, weather, historical speed)
    - LSTM layers: Capture temporal dependencies
    - Fully connected layers: Final prediction
    - Dropout: Prevent overfitting
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(TrafficLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last time step's output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out


def generate_mock_traffic_data(num_samples=50000):
    """
    Generate realistic mock traffic data for training
    Simulates traffic patterns with time-of-day and weather effects
    """
    print("Generating mock traffic data...")
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(num_samples):
        timestamp = start_date + timedelta(minutes=i * 5)
        
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Base speed (km/h)
        base_speed = 50
        
        # Time of day effects (rush hours)
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Morning and evening rush
            time_factor = 0.6  # 40% slower
        elif 22 <= hour or hour <= 5:  # Night time
            time_factor = 1.2  # 20% faster
        else:
            time_factor = 1.0
        
        # Weekend effect
        weekend_factor = 1.1 if day_of_week >= 5 else 1.0
        
        # Weather effect (random)
        weather = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])  # Clear, Rain, Snow
        weather_factor = 1.0 if weather == 0 else (0.8 if weather == 1 else 0.6)
        
        # Road segment (0-99)
        road_segment = np.random.randint(0, 100)
        
        # Calculate speed with some randomness
        speed = base_speed * time_factor * weekend_factor * weather_factor
        speed += np.random.normal(0, 5)  # Add noise
        speed = max(10, min(120, speed))  # Clip to realistic range
        
        data.append({
            'timestamp': timestamp,
            'hour': hour,
            'day_of_week': day_of_week,
            'weather': weather,
            'road_segment': road_segment,
            'speed': speed
        })
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} traffic samples")
    return df


def create_sequences(data, seq_length=12):
    """
    Create sequences for LSTM training
    seq_length: number of time steps to look back (e.g., 12 = 1 hour with 5-min intervals)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # Input sequence: past seq_length time steps
        X.append(data[i:i+seq_length])
        # Target: next time step's speed
        y.append(data[i+seq_length, -1])  # Last column is speed
    
    return np.array(X), np.array(y)


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """Train the LSTM model"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nTraining on {device}...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions.squeeze(), y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_traffic_model.pth')
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    print("=" * 60)
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses


def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Traffic Prediction Model - Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'")


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model performance on test set"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            predictions.extend(pred.cpu().numpy().flatten())
            actuals.extend(y_batch.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean Squared Error (MSE):  {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} km/h")
    print(f"Mean Absolute Error (MAE): {mae:.4f} km/h")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("=" * 60)
    
    return predictions, actuals


def save_model_for_production(model, scaler, seq_length, input_size):
    """Save model and metadata for production use"""
    
    # Save model architecture and weights
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'seq_length': seq_length
    }, 'traffic_model_production.pth')
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'traffic_scaler.pkl')
    
    # Save metadata
    metadata = {
        'model_type': 'LSTM',
        'input_features': ['hour', 'day_of_week', 'weather', 'road_segment', 'speed'],
        'seq_length': seq_length,
        'input_size': input_size,
        'created_at': datetime.now().isoformat(),
        'description': 'Traffic speed prediction model for autonomous EVs'
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✓ Model saved for production:")
    print("  - traffic_model_production.pth")
    print("  - traffic_scaler.pkl")
    print("  - model_metadata.json")


def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("TRAFFIC PREDICTION MODEL - TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Generate or load data
    df = generate_mock_traffic_data(num_samples=50000)
    
    # 2. Feature engineering
    features = ['hour', 'day_of_week', 'weather', 'road_segment', 'speed']
    data = df[features].values
    
    # 3. Normalize data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    
    # 4. Create sequences
    seq_length = 12  # Look back 12 time steps (1 hour)
    X, y = create_sequences(data_normalized, seq_length)
    
    print(f"\nDataset shape:")
    print(f"  X: {X.shape} (samples, time_steps, features)")
    print(f"  y: {y.shape} (samples,)")
    
    # 5. Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)  # 0.176 of 0.85 = 0.15
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # 6. Create DataLoaders
    batch_size = 64
    train_dataset = TrafficDataset(X_train, y_train)
    val_dataset = TrafficDataset(X_val, y_val)
    test_dataset = TrafficDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 7. Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X.shape[2]  # Number of features
    
    model = TrafficLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 8. Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=50, lr=0.001, device=device
    )
    
    # 9. Plot training history
    plot_training_history(train_losses, val_losses)
    
    # 10. Load best model and evaluate
    checkpoint = torch.load('best_traffic_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    predictions, actuals = evaluate_model(model, test_loader, device)
    
    # 11. Save for production
    save_model_for_production(model, scaler, seq_length, input_size)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()