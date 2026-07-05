import os
import joblib
from sklearn.svm import SVC
from preprocessing import DataPreprocessor

if __name__ == "__main__":
    preprocessor = DataPreprocessor("breast-cancer.csv")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()

    model = SVC(probability=True)
    model.fit(X_train, y_train)

    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/model.pkl")
    joblib.dump(preprocessor.scaler, "../models/scaler.pkl")
    
    print("✅ Best model and scaler successfully saved to '../models'")
