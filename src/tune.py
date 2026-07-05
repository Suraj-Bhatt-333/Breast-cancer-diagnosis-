import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from preprocessing import DataPreprocessor

if __name__ == "__main__":
    preprocessor = DataPreprocessor("breast-cancer.csv")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()

    print("Starting Grid Search...")
    param_grid = {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
    
    grid = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print("Best Parameters:", grid.best_params_)
    
    os.makedirs("../models", exist_ok=True)
    joblib.dump(grid.best_estimator_, "../models/model.pkl")
    joblib.dump(preprocessor.scaler, "../models/scaler.pkl")
    print("✅ Tuned model saved.")
