# Setup Instructions
## Intelligent Route Optimization for Autonomous Electric Vehicles

Complete guide to set up and run the AEV Route Optimizer system locally.

---

## Prerequisites

Before starting, ensure you have the following installed:

- **Python 3.8 or higher** ([Download](https://www.python.org/downloads/))
- **Node.js 14+ and npm** ([Download](https://nodejs.org/))
- **Git** (optional, for cloning)
- **8GB RAM minimum** (for ML model training)
- **Internet connection** (for OpenStreetMap data)

### Check Installations:
```bash
python --version    # Should be 3.8+
node --version      # Should be 14+
npm --version       # Should be 6+
```

---

## Step 1: Project Setup

### Option A: If you have the files already
Navigate to the project directory:
```bash
cd AEV-Route-Optimizer
```

### Option B: Create structure from scratch
```bash
mkdir AEV-Route-Optimizer
cd AEV-Route-Optimizer

# Create directory structure
mkdir -p ml_model route_optimizer backend frontend/src frontend/public
```

---

## Step 2: Machine Learning Model Setup

### 2.1 Install Python Dependencies
```bash
cd ml_model

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2.2 Train the Traffic Prediction Model
```bash
# This will take 5-10 minutes
python train_traffic_model.py
```

**Expected Output:**
```
=========================================================
TRAFFIC PREDICTION MODEL - TRAINING PIPELINE
=========================================================

Generating mock traffic data...
Generated 50000 traffic samples

Dataset shape:
  X: (49988, 12, 5) (samples, time_steps, features)
  y: (49988,) (samples,)

...

Training completed! Best validation loss: X.XXXX

MODEL EVALUATION RESULTS
=========================================================
Mean Squared Error (MSE):  X.XXXX
Root Mean Squared Error (RMSE): X.XX km/h
Mean Absolute Error (MAE): X.XX km/h
Mean Absolute Percentage Error (MAPE): X.XX%
=========================================================

✓ Model saved for production:
  - traffic_model_production.pth
  - traffic_scaler.pkl
  - model_metadata.json
```

### 2.3 Verify Model Training
```bash
# Test the inference
python traffic_predictor.py
```

---

## Step 3: Backend API Setup

### 3.1 Install Backend Dependencies
```bash
# Go to backend directory
cd ../backend

# If not using the same virtual environment:
# python -m venv venv
# source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 3.2 Test Route Optimizer
```bash
cd ../route_optimizer

# Test route optimization algorithms
python route_optimizer.py
```

**Expected Output:**
```
=========================================================
TESTING ROUTE OPTIMIZATION ALGORITHMS
=========================================================

Creating mock city network (10x10 grid)...
  Nodes: 100
  Edges: 360

Finding routes from node 0 to node 99
------------------------------------------------------------

Fastest Route (A* Algorithm):
  Path: [0, 1, 2, 3, 4]...[95, 96, 97, 98, 99]
  Total nodes: 19
  Distance: 18.00 km
  Time: XX.X minutes
  Energy: X.XX kWh

...

A* is X.XXx faster
```

### 3.3 Start the Backend Server
```bash
cd ../backend

# Run the FastAPI server
python main.py
```

**Expected Output:**
```
=========================================================
INITIALIZING AEV ROUTE OPTIMIZATION SYSTEM
=========================================================

1. Loading traffic prediction model...
✓ Traffic predictor loaded

2. Loading road network...
✓ Road network loaded

3. Initializing route optimizer...
✓ Route optimizer ready

=========================================================
SYSTEM READY - API Server Started
=========================================================

INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Keep this terminal running!**

### 3.4 Test API Endpoints
Open a new terminal and test:

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test network stats
curl http://localhost:8000/api/network/stats
```

Or visit in browser:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

## Step 4: Frontend Setup

### 4.1 Install Node Dependencies
```bash
# Open a new terminal
cd frontend

# Install dependencies (this may take a few minutes)
npm install
```

### 4.2 Start Development Server
```bash
npm start
```

**Expected Output:**
```
Compiled successfully!

You can now view aev-route-optimizer-frontend in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000

Note that the development build is not optimized.
To create a production build, use npm run build.
```

**The application will automatically open in your browser!**

---

## Step 5: Using the Application

### 5.1 Access the Interface
Open your browser to: **http://localhost:3000**

### 5.2 Test Route Optimization

1. **Set Start Location**
   - Default: Lat 28.6139, Lon 77.2090 (Delhi, India)
   - Or enter your own coordinates

2. **Set Destination**
   - Default: Lat 28.6500, Lon 77.2500
   - Adjust as needed

3. **Configure Vehicle**
   - Battery Capacity: 60 kWh
   - Efficiency: 0.2 kWh/km
   - Current Charge: 100%

4. **Select Optimization**
   - Objective: Balanced (recommended)
   - Algorithm: A* (faster)
   - Weather: Clear

5. **Click "Optimize Route"**
   - Wait 1-2 seconds for computation
   - Route will appear on map
   - Statistics will update

### 5.3 Interpret Results

**Map Display:**
- Blue line: Optimized route
- Green marker: Start location
- Red marker: Destination

**Statistics Cards:**
- 📏 Total Distance (km)
- ⏱️ Travel Time (minutes)
- ⚡ Energy Used (kWh)
- 🔋 Battery Remaining (%)
- 🎯 ETA (estimated time of arrival)
- 💻 Computation Time (ms)

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
# Reinstall Python dependencies
pip install -r requirements.txt

# Reinstall Node dependencies
cd frontend
rm -rf node_modules
npm install
```

### Issue: "Port 8000 already in use"

**Solution:**
```bash
# Find and kill process on port 8000
# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# On macOS/Linux:
lsof -ti:8000 | xargs kill -9
```

### Issue: "Model files not found"

**Solution:**
```bash
# Re-run model training
cd ml_model
python train_traffic_model.py
```

### Issue: "CORS errors" in browser

**Solution:**
- Make sure backend is running on port 8000
- Check frontend proxy in package.json
- Try clearing browser cache

### Issue: Map not displaying

**Solution:**
- Check internet connection (needs OpenStreetMap tiles)
- Verify Leaflet CSS is loaded
- Check browser console for errors

### Issue: "Cannot connect to backend"

**Solution:**
```bash
# Verify backend is running
curl http://localhost:8000/health

# Check if correct port
# Backend should be on 8000
# Frontend should be on 3000
```

---

## Running Individual Components

### Test ML Model Only:
```bash
cd ml_model
python train_traffic_model.py  # Train
python traffic_predictor.py     # Test inference
```

### Test Route Optimizer Only:
```bash
cd route_optimizer
python route_optimizer.py  # Test algorithms
python osm_loader.py       # Test OSM loading
```

### Run Backend Only:
```bash
cd backend
python main.py
# Visit http://localhost:8000/docs for API testing
```

### Run Frontend Only:
```bash
cd frontend
npm start
# Note: Needs backend running for full functionality
```

---

## Production Deployment

### Build Frontend for Production:
```bash
cd frontend
npm run build
# Creates optimized build in frontend/build/
```

### Run Backend in Production:
```bash
cd backend
# Using gunicorn (install first: pip install gunicorn)
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Environment Variables:
Create a `.env` file for production settings:
```bash
# Backend
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=../ml_model/traffic_model_production.pth

# Frontend
REACT_APP_API_URL=https://your-api-domain.com
```

---

## Testing the System

### Test Case 1: Short Route
```
Start: 28.6139, 77.2090
End:   28.6200, 77.2150
Expected: ~1-2 km route, < 5 min
```

### Test Case 2: Different Objectives
```
Test all four objectives:
1. Fastest - Should prioritize highways
2. Shortest - May use smaller roads
3. Energy - Avoid hills if present
4. Balanced - Compromise between all
```

### Test Case 3: Weather Impact
```
Same route with different weather:
- Clear: Normal speeds
- Rain: ~20% slower
- Snow: ~40% slower
```

### Test Case 4: Low Battery
```
Set current charge to 20%
Check if energy consumption stays within limits
```

---

## Next Steps

1. **Customize for Your Area:**
   - Update default coordinates
   - Load OSM data for your region
   - Adjust speed limits for local roads

2. **Enhance ML Model:**
   - Add real traffic data
   - Train on historical patterns
   - Implement ensemble methods

3. **Add Features:**
   - Charging station waypoints
   - Multi-stop routing
   - Route comparison tool
   - Export routes to GPS

4. **Integrate Real APIs:**
   - Google Maps Traffic
   - Weather APIs
   - EV charging networks

---

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review console logs (browser and terminal)
3. Verify all dependencies are installed
4. Ensure all servers are running

---

## Quick Start Summary

```bash
# Terminal 1: Backend
cd backend
pip install -r requirements.txt
python main.py

# Terminal 2: Frontend
cd frontend
npm install
npm start

# Browser
Open http://localhost:3000
Click "Optimize Route"
```

**That's it! You're ready to optimize routes for autonomous EVs! 🚗⚡🗺️**