import joblib
import os
from sklearn.svm import SVC
from preprocessing import DataPreprocessor

# Step 4: Load the data
# (Using the OOP DataPreprocessor we created earlier)
preprocessor = DataPreprocessor("breast-cancer.csv")
X_train, X_test, y_train, y_test = preprocessor.prepare_data()
scaler = preprocessor.scaler

# Step 5: Train the final model
# Setting probability=True so we can use predict_proba() in Streamlit
model = SVC(probability=True)
model.fit(X_train, y_train)

# Create models directory if it doesn't exist
models_dir = "../models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Step 6: Save Model
joblib.dump(model, os.path.join(models_dir, "model.pkl"))

# Step 7: Save Scaler
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

print("Congratulations 🎉")
print("You have now created and saved a trained ML model.")
