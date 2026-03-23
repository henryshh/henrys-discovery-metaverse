import json
import sqlite3
import numpy as np
import uuid
from pathlib import Path

def generate_curve(anomalous=False, fault_type=None):
    angle = np.linspace(0, 100, 200)
    # Basic sigmoidal curve for torque
    torque = 10 / (1 + np.exp(-0.1 * (angle - 50)))
    
    # Add noise
    torque += np.random.normal(0, 0.05, len(torque))
    
    # Inject fault
    if anomalous:
        if fault_type == "soft_joint":
            # Lower stiffness (stretched out)
            torque = 10 / (1 + np.exp(-0.05 * (angle - 60)))
        elif fault_type == "cross_thread":
            # High initial torque
            torque[:20] += 2.0
            
    return angle.tolist(), torque.tolist()

def main():
    db_path = Path("data/b8dcf59d-d931-4f10-a3c2-d2fe671a93b7.db")
    if not db_path.exists():
        # Create it if it doesn't exist (it should, but just in case)
        print(f"DB not found at {db_path}, but will attempt connection...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    dataset_id = "sota_mock_v1"
    
    # Insert Curves
    curves = []
    for i in range(20):
        fault = None
        is_ok = "OK"
        if i == 18: 
            fault = "soft_joint"
            is_ok = "NOK"
        elif i == 19:
            fault = "cross_thread"
            is_ok = "NOK"
            
        angle, torque = generate_curve(anomalous=(fault is not None), fault_type=fault)
        
        curve_data = {
            "resultNumber": 1000 + i,
            "vin": f"VIN{i:04d}",
            "torque": torque,
            "angle": angle,
            "current": (np.array(torque) * 0.5).tolist(), # Current as proxy for torque
            "marker": [0] * len(torque),
            "report": is_ok,
            "pointDuration": 15000,
            "pointIndices": list(range(len(torque)))
        }
        
        cursor.execute(
            "INSERT INTO curves (dataset_id, data) VALUES (?, ?)",
            (dataset_id, json.dumps(curve_data))
        )
    
    conn.commit()
    conn.close()
    print(f"Imported 20 SOTA curves into {dataset_id}")

if __name__ == "__main__":
    main()
