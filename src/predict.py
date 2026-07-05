import joblib
import numpy as np
import os

# Define paths (relative to where the script is run)
MODEL_PATH = "../models/model.pkl"
SCALER_PATH = "../models/scaler.pkl"

# Load files globally so Streamlit can use them easily
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    model = None
    scaler = None

def predict(data):
    """
    Predicts the class and the probability (confidence).
    data: list or array of features
    """
    if model is None or scaler is None:
         raise FileNotFoundError("Model or scaler not found. Run save_model.py first.")

    # Convert data into a 2D numpy array (1 row, -1 columns)
    data = np.array(data).reshape(1, -1)
    
    # Scale the input data
    data = scaler.transform(data)
    
    # Get the prediction (0 or 1)
    prediction = model.predict(data)[0]
    
    # Get the maximum probability (confidence score)
    probability = model.predict_proba(data).max()
    
    return prediction, probability

if __name__ == "__main__":
    # Test it using the first patient from the dataset (30 features)
    sample = [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 
        0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 
        0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 
        0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
    
    try:
        prediction, probability = predict(sample)
        
        print("Prediction (Raw):", prediction)
        print(f"Confidence (Probability): {probability * 100:.2f}%\n")
        
        if prediction == 1:
            print("Diagnosis: Malignant")
        else:
            print("Diagnosis: Benign")
            
    except Exception as e:
        print("Error during prediction:", e)
