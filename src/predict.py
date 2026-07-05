import joblib
import numpy as np
import os

if __name__ == "__main__":
    if os.path.exists("../models/model.pkl") and os.path.exists("../models/scaler.pkl"):
        model = joblib.load("../models/model.pkl")
        scaler = joblib.load("../models/scaler.pkl")
        
        # Test Sample
        sample = np.array([
            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 
            0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 
            0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 
            0.6656, 0.7119, 0.2654, 0.4601, 0.1189
        ]).reshape(1, -1)
        
        scaled_sample = scaler.transform(sample)
        prediction = model.predict(scaled_sample)[0]
        
        print("Diagnosis:", "Malignant" if prediction == 1 else "Benign")
        print(f"Confidence: {model.predict_proba(scaled_sample).max() * 100:.2f}%")
    else:
        print("Run save_model.py first to generate the models.")
