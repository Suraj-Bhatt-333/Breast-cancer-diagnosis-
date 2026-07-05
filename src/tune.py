import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from preprocessing import DataPreprocessor

class HyperparameterTuner:
    """
    Handles hyperparameter tuning using GridSearchCV and model saving.
    """
    def __init__(self):
        # Define parameter grid for SVM
        self.param_grid = {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
        self.model = SVC()
        
    def tune(self, X_train, y_train):
        """
        Runs GridSearch to find the best hyperparameters.
        """
        print("Starting Grid Search...")
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )
        
        grid.fit(X_train, y_train)
        
        print("\nBest Parameters found:")
        print(grid.best_params_)
        
        print(f"\nBest Cross-validation Accuracy: {grid.best_score_:.4f}")
        return grid.best_estimator_

    def save_model_and_scaler(self, best_model, scaler, models_dir="../models"):
        """
        Saves the best model and the data scaler using joblib.
        """
        # Ensure the directory exists
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        model_path = os.path.join(models_dir, "model.pkl")
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"\n✅ Model and scaler successfully saved to '{models_dir}/'")


if __name__ == "__main__":
    # 1. Load Data
    preprocessor = DataPreprocessor("breast-cancer.csv")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    
    # 2. Tune Model
    tuner = HyperparameterTuner()
    best_svc_model = tuner.tune(X_train, y_train)
    
    # 3. Save Model and Scaler (Phase 6)
    # Using 'models' in the project root directory
    tuner.save_model_and_scaler(best_svc_model, preprocessor.scaler, models_dir="../models")
