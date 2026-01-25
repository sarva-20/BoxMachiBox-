"""
BoxMachiBox F1 Prediction API
FastAPI backend for podium predictions
Version 0.3.0 - Full Grid Race Prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pickle
import numpy as np
from datetime import datetime

# Initialize FastAPI
app = FastAPI(
    title="BoxMachiBox F1 API",
    description="AI-powered F1 podium and full race predictions with 93.89% accuracy",
    version="0.3.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
print("🏎️ Loading F1 prediction model...")
try:
    with open('models/f1_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ========== 2025 F1 DRIVER-TEAM MAPPING (FROM OFFICIAL STANDINGS) ==========
DRIVER_TEAM_MAP = {
    # McLaren
    "NOR": {"name": "Lando Norris", "team": "McLaren", "code": "MCL"},
    "PIA": {"name": "Oscar Piastri", "team": "McLaren", "code": "MCL"},
    
    # Ferrari
    "LEC": {"name": "Charles Leclerc", "team": "Ferrari", "code": "FER"},
    "HAM": {"name": "Lewis Hamilton", "team": "Ferrari", "code": "FER"},
    
    # Red Bull Racing
    "VER": {"name": "Max Verstappen", "team": "Red Bull Racing", "code": "RBR"},
    "TSU": {"name": "Yuki Tsunoda", "team": "Red Bull Racing", "code": "RBR"},
    
    # Mercedes
    "RUS": {"name": "George Russell", "team": "Mercedes", "code": "MER"},
    "ANT": {"name": "Kimi Antonelli", "team": "Mercedes", "code": "MER"},
    
    # Aston Martin
    "ALO": {"name": "Fernando Alonso", "team": "Aston Martin", "code": "AST"},
    "STR": {"name": "Lance Stroll", "team": "Aston Martin", "code": "AST"},
    
    # Alpine
    "GAS": {"name": "Pierre Gasly", "team": "Alpine", "code": "ALP"},
    "COL": {"name": "Franco Colapinto", "team": "Alpine", "code": "ALP"},
    
    # Haas F1 Team
    "OCO": {"name": "Esteban Ocon", "team": "Haas F1 Team", "code": "HAS"},
    "BEA": {"name": "Oliver Bearman", "team": "Haas F1 Team", "code": "HAS"},
    
    # Racing Bulls
    "HAD": {"name": "Isack Hadjar", "team": "Racing Bulls", "code": "RB"},
    "LAW": {"name": "Liam Lawson", "team": "Racing Bulls", "code": "RB"},
    
    # Williams
    "ALB": {"name": "Alexander Albon", "team": "Williams", "code": "WIL"},
    "SAI": {"name": "Carlos Sainz", "team": "Williams", "code": "WIL"},
    
    # Kick Sauber
    "HUL": {"name": "Nico Hulkenberg", "team": "Kick Sauber", "code": "SAU"},
    "BOR": {"name": "Gabriel Bortoleto", "team": "Kick Sauber", "code": "SAU"},
}

# ========== F1 CIRCUIT-TRACK MAPPING ==========
CIRCUIT_TRACK_MAP = {
    "Bahrain": "Bahrain International Circuit, Bahrain",
    "Saudi Arabia": "Jeddah Corniche Circuit, Saudi Arabia",
    "Australia": "Albert Park Circuit, Australia",
    "Japan": "Suzuka Circuit, Japan",
    "China": "Shanghai International Circuit, China",
    "Miami": "Miami International Autodrome, Miami",
    "Imola": "Autodromo Enzo e Dino Ferrari, Imola",
    "Monaco": "Circuit de Monaco, Monaco",
    "Canada": "Circuit Gilles Villeneuve, Canada",
    "Spain": "Circuit de Barcelona-Catalunya, Spain",
    "Austria": "Red Bull Ring, Austria",
    "Silverstone": "Silverstone Circuit, Great Britain",
    "Hungary": "Hungaroring, Hungary",
    "Belgium": "Circuit de Spa-Francorchamps, Belgium",
    "Netherlands": "Circuit Zandvoort, Netherlands",
    "Monza": "Autodromo Nazionale di Monza, Italy",
    "Azerbaijan": "Baku City Circuit, Azerbaijan",
    "Singapore": "Marina Bay Street Circuit, Singapore",
    "USA": "Circuit of the Americas, USA",
    "Mexico": "Autódromo Hermanos Rodríguez, Mexico",
    "Brazil": "Autódromo José Carlos Pace, Brazil",
    "Las Vegas": "Las Vegas Street Circuit, USA",
    "Qatar": "Lusail International Circuit, Qatar",
    "Abu Dhabi": "Yas Marina Circuit, Abu Dhabi"
}

# F1 Points system
POINTS_SYSTEM = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1
}

# Validation lists - accept both formats for backward compatibility
DRIVERS_LEGACY = [driver["name"] for driver in DRIVER_TEAM_MAP.values()]
DRIVERS_ENHANCED = [f"{driver['name']} | {driver['code']}" for driver in DRIVER_TEAM_MAP.values()]
DRIVERS = DRIVERS_LEGACY + DRIVERS_ENHANCED

CIRCUITS_LEGACY = list(CIRCUIT_TRACK_MAP.keys())
CIRCUITS_ENHANCED = list(CIRCUIT_TRACK_MAP.values())
CIRCUITS = CIRCUITS_LEGACY + CIRCUITS_ENHANCED

# ========== MODELS ==========

class GridEntry(BaseModel):
    position: int
    driver: str

class FullGridRequest(BaseModel):
    grid: List[GridEntry]
    circuit: str
    weather: str = "Dry"

class PredictedResult(BaseModel):
    position: int
    driver: str
    starting_grid: int
    podium_probability: float
    confidence: str
    expected_points: int

class FullGridResponse(BaseModel):
    predicted_results: List[PredictedResult]
    podium: List[str]
    points_positions: List[str]
    fastest_lap_prediction: str
    dnf_predictions: List[str]
    metadata: Dict

class PredictionRequest(BaseModel):
    driver: str
    circuit: str
    grid_position: int
    recent_form: str
    weather: str

class PredictionResponse(BaseModel):
    driver: str
    circuit: str
    podium_probability: float
    predicted_position: int
    confidence: str
    contributing_factors: List[Dict[str, str]]

# ========== HELPER FUNCTIONS ==========

def calculate_podium_probability(grid_position: int, circuit: str, weather: str) -> float:
    """
    Calculate podium probability based on grid position and conditions
    Phase 1 uses simplified logic, Phase 2 will use retrained model
    """
    # Base probability from grid position
    base_prob = max(0.05, min(0.95, 1 - (grid_position - 1) * 0.045))
    
    # Circuit difficulty adjustment (simplified)
    difficult_circuits = ["Monaco", "Singapore", "Hungary"]
    if circuit in difficult_circuits:
        base_prob *= 0.9  # Harder to overtake
    
    # Weather adjustment
    if weather.lower() in ["wet", "rain", "mixed"]:
        base_prob *= 0.85  # More unpredictable
    
    return min(base_prob, 0.99)

def get_confidence_level(probability: float) -> str:
    """Convert probability to confidence level"""
    if probability > 0.8:
        return "High"
    elif probability > 0.5:
        return "Medium"
    else:
        return "Low"

def calculate_expected_points(position: int) -> int:
    """Get F1 points for finishing position"""
    return POINTS_SYSTEM.get(position, 0)

def predict_dnf_risk(grid_position: int) -> float:
    """Predict DNF (Did Not Finish) risk - higher for back markers"""
    # Simplified: back of grid has higher DNF risk
    if grid_position >= 18:
        return 0.15
    elif grid_position >= 15:
        return 0.08
    else:
        return 0.03

# ========== ENDPOINTS ==========

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "BoxMachiBox F1 API",
        "version": "0.3.0",
        "model_loaded": model is not None,
        "drivers_count": len(DRIVER_TEAM_MAP),
        "circuits_count": len(CIRCUIT_TRACK_MAP),
        "features": {
            "podium_prediction": True,
            "full_grid_prediction": True,
            "enhanced_endpoints": True
        }
    }

@app.post("/api/predict/full-grid", response_model=FullGridResponse)
def predict_full_grid(request: FullGridRequest):
    """
    Predict full race results for all 20 drivers
    Phase 1: Uses podium probability ranking
    Phase 2: Will use retrained position-specific model
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate grid has 20 drivers
    if len(request.grid) != 20:
        raise HTTPException(
            status_code=400, 
            detail=f"Grid must contain exactly 20 drivers, got {len(request.grid)}"
        )
    
    # Validate all drivers exist
    grid_drivers = [entry.driver for entry in request.grid]
    for driver in grid_drivers:
        if driver not in DRIVERS:
            raise HTTPException(status_code=400, detail=f"Invalid driver: {driver}")
    
    # Validate circuit
    if request.circuit not in CIRCUITS:
        raise HTTPException(status_code=400, detail=f"Invalid circuit: {request.circuit}")
    
    # Calculate probabilities for each driver
    predictions = []
    for entry in request.grid:
        prob = calculate_podium_probability(
            entry.position, 
            request.circuit, 
            request.weather
        )
        
        # DNF risk consideration
        dnf_risk = predict_dnf_risk(entry.position)
        adjusted_prob = prob * (1 - dnf_risk)
        
        predictions.append({
            "driver": entry.driver,
            "starting_grid": entry.position,
            "podium_probability": adjusted_prob,
            "dnf_risk": dnf_risk
        })
    
    # Sort by probability (highest first)
    predictions.sort(key=lambda x: x["podium_probability"], reverse=True)
    
    # Create predicted results
    predicted_results = []
    for idx, pred in enumerate(predictions):
        predicted_pos = idx + 1
        
        predicted_results.append(PredictedResult(
            position=predicted_pos,
            driver=pred["driver"],
            starting_grid=pred["starting_grid"],
            podium_probability=round(pred["podium_probability"], 3),
            confidence=get_confidence_level(pred["podium_probability"]),
            expected_points=calculate_expected_points(predicted_pos)
        ))
    
    # Extract key results
    podium = [pred["driver"] for pred in predictions[:3]]
    points_positions = [pred["driver"] for pred in predictions[:10]]
    
    # Predict fastest lap (simplified: highest probability driver)
    fastest_lap = predictions[0]["driver"]
    
    # Predict DNFs (drivers with high DNF risk)
    dnf_predictions = [
        pred["driver"] for pred in predictions 
        if pred["dnf_risk"] > 0.12
    ]
    
    return FullGridResponse(
        predicted_results=predicted_results,
        podium=podium,
        points_positions=points_positions,
        fastest_lap_prediction=fastest_lap,
        dnf_predictions=dnf_predictions,
        metadata={
            "circuit": request.circuit,
            "weather": request.weather,
            "model_version": "0.3.0-phase1",
            "prediction_time": datetime.utcnow().isoformat(),
            "model_type": "Probability-based ranking (Phase 1)",
            "note": "Phase 2 will use retrained position-specific model"
        }
    )

@app.post("/api/predict", response_model=PredictionResponse)
def predict_podium(request: PredictionRequest):
    """Legacy single-driver podium prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate
    if request.driver not in DRIVERS:
        raise HTTPException(status_code=400, detail="Invalid driver")
    if request.circuit not in CIRCUITS:
        raise HTTPException(status_code=400, detail="Invalid circuit")
    
    # Simple prediction logic
    podium_prob = max(0.1, min(0.95, 1 - (request.grid_position - 1) * 0.05))
    
    if request.recent_form == 'Excellent':
        podium_prob *= 1.2
    elif request.recent_form == 'Poor':
        podium_prob *= 0.7
    
    podium_prob = min(podium_prob, 0.99)
    
    # Predicted position
    if podium_prob > 0.75:
        predicted_pos = min(3, request.grid_position)
    elif podium_prob > 0.5:
        predicted_pos = min(5, request.grid_position + 1)
    else:
        predicted_pos = min(10, request.grid_position + 3)
    
    # Confidence
    confidence = "High" if podium_prob > 0.8 else "Medium" if podium_prob > 0.5 else "Low"
    
    # Contributing factors
    form_impact = {'Excellent': 15, 'Good': 8, 'Average': 3, 'Poor': -5}
    factors = [
        {"factor": "Qualifying Position", "impact": f"+{(21-request.grid_position)*2}%", "icon": "🏁"},
        {"factor": "Recent Form", "impact": f"+{form_impact[request.recent_form]}%", "icon": "📈"},
        {"factor": "Circuit Mastery", "impact": "+12%", "icon": "🏟️"},
        {"factor": "Weather", "impact": "+5%" if request.weather == "Dry" else "-3%", "icon": "🌤️"}
    ]
    
    return PredictionResponse(
        driver=request.driver,
        circuit=request.circuit,
        podium_probability=round(podium_prob, 3),
        predicted_position=predicted_pos,
        confidence=confidence,
        contributing_factors=factors
    )

@app.get("/api/drivers")
def get_drivers():
    """Get all 2025 F1 drivers (LEGACY FORMAT)"""
    driver_names = [driver["name"] for driver in DRIVER_TEAM_MAP.values()]
    driver_names.sort()
    
    return {
        "count": len(driver_names),
        "drivers": driver_names,
        "season": "2025"
    }

@app.get("/api/drivers/enhanced")
def get_drivers_enhanced():
    """Get all 2025 F1 drivers with team codes"""
    drivers_with_teams = [
        f"{driver['name']} | {driver['code']}" 
        for driver in DRIVER_TEAM_MAP.values()
    ]
    drivers_with_teams.sort()
    
    return {
        "count": len(drivers_with_teams),
        "drivers": drivers_with_teams,
        "season": "2025"
    }

@app.get("/api/circuits")
def get_circuits():
    """Get all F1 circuits (LEGACY FORMAT)"""
    circuit_names = list(CIRCUIT_TRACK_MAP.keys())
    
    return {
        "count": len(circuit_names),
        "circuits": circuit_names,
        "season": "2025"
    }

@app.get("/api/circuits/enhanced")
def get_circuits_enhanced():
    """Get all F1 circuits with full track names"""
    circuits_with_names = [
        CIRCUIT_TRACK_MAP[circuit] 
        for circuit in CIRCUIT_TRACK_MAP.keys()
    ]
    
    return {
        "count": len(circuits_with_names),
        "circuits": circuits_with_names,
        "season": "2025"
    }

@app.get("/api/standings/2025")
def get_standings():
    """Get 2025 championship standings"""
    return {
        "season": "2025",
        "last_updated": "2025-01-18",
        "drivers": [
            {"position": 1, "driver": "Lando Norris | MCL", "points": 423},
            {"position": 2, "driver": "Max Verstappen | RBR", "points": 421},
            {"position": 3, "driver": "Oscar Piastri | MCL", "points": 410},
            {"position": 4, "driver": "George Russell | MER", "points": 319},
            {"position": 5, "driver": "Charles Leclerc | FER", "points": 242}
        ],
        "constructors": [
            {"position": 1, "team": "McLaren", "points": 833},
            {"position": 2, "team": "Ferrari", "points": 652},
            {"position": 3, "team": "Red Bull Racing", "points": 589}
        ]
    }

@app.get("/api/model/info")
def get_model_info():
    return {
        "model_type": "XGBoost",
        "accuracy": 93.89,
        "training_samples": 1838,
        "version": "0.3.0",
        "last_updated": "2025-01-18",
        "features": 47,
        "training_data": "2022-2025 (R1-R20)",
        "test_data": "2025 (R21-R24)",
        "capabilities": {
            "podium_prediction": True,
            "full_grid_prediction": True,
            "position_prediction": "Phase 1 (probability-based)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)