import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data_path="breast-cancer.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()

    def load_data(self):
        return pd.read_csv(self.data_path)

    def clean_data(self, df):
        if "id" in df.columns:
            df = df.drop("id", axis=1)
        if "diagnosis" in df.columns:
            df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})
        return df

    def split_data(self, df, target_col="diagnosis", test_size=0.2, random_state=42):
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def prepare_data(self):
        """Executes the full preprocessing pipeline and returns scaled data."""
        df = self.load_data()
        df = self.clean_data(df)
        X_train, X_test, y_train, y_test = self.split_data(df)
        X_train_scaled, X_test_scaled = self.scale_data(X_train, X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

