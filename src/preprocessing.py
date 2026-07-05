import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data_path="breast-cancer.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()

    def prepare_data(self):
        """Loads, cleans, splits, and scales data efficiently."""
        df = pd.read_csv(self.data_path)
        
        # Clean
        if "id" in df.columns:
            df = df.drop("id", axis=1)
        if "diagnosis" in df.columns:
            df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})
            
        # Split
        X = df.drop("diagnosis", axis=1)
        y = df["diagnosis"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
