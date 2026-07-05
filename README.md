# 🩺 Breast Cancer Diagnosis using Machine Learning

### CODTECH Internship - Machine Learning Project

**Company** : CODTECH IT SOLUTIONS

**Name** : Suraj Bhatt

**Intern ID** : CITS6462

**Domain** : Machine Learning

**Duration** : 4 Weeks

**Mentor** : Neela Santhosh Kumar

---

## 🌐 Live Application

https://breast-cancer-diagnosis-using-ml.streamlit.app/

---

## 📂 Dataset

Kaggle Breast Cancer Dataset

https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

---

# 📌 Problem Statement

Breast cancer is one of the most common cancers among women worldwide. its diagnosis plays a crucial role in increasing the chances of successful treatment.

The objective of this project was to develop a Machine Learning model capable of predicting whether a breast tumor is **Benign (Non-Cancerous)** or **Malignant (Cancerous)** based on several medical measurements extracted from digitized images of breast cell nuclei.

Instead of stopping at building a machine learning model, the goal was to create a complete end-to-end ML application that takes user input through a web interface and predicts the diagnosis in real time.

---

# 🚀 Project Overview

This project marks one of my first complete Machine Learning projects where I followed the entire workflow of developing an ML application, starting from a raw dataset and ending with a deployed web application.

Although the dataset itself is relatively simple, the project helped me understand the complete lifecycle of a Machine Learning project, including data analysis, preprocessing, model selection, evaluation, deployment, and version control.

Rather than training only a single algorithm, I compared multiple Machine Learning models, evaluated their performance, selected the best-performing model, and finally deployed it using Streamlit.

This project helped me move beyond tutorial-based coding and understand how real-world ML projects are structured.

---

# 🛠️ Workflow Followed

### 1. Dataset Exploration (EDA)

- Loaded the dataset using Pandas
- Explored the dataset structure
- Checked feature data types
- Checked missing values
- Checked duplicate records
- Studied class distribution
- Generated correlation heatmap
- Understood feature relationships

---

### 2. Data Preprocessing

- Removed unnecessary ID column
- Encoded diagnosis labels
- Split dataset into training and testing sets
- Applied feature scaling using StandardScaler
- Prepared data for machine learning models

---

### 3. Model Training

Instead of relying on a single algorithm, I trained multiple classification models.

The models compared were:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gaussian Naive Bayes
- Gradient Boosting Classifier

---

### 4. Model Evaluation

Each model was evaluated on the testing dataset.

The achieved accuracies were:

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 96.49% |
| KNN | 95.61% |
| Decision Tree | 92.11% |
| Random Forest | 96.49% |
| Support Vector Machine | **97.37%** |
| Gaussian Naive Bayes | 92.11% |
| Gradient Boosting | 96.49% |

Among all the models, **Support Vector Machine (SVM)** produced the highest accuracy of **97.37%**, making it the final model used for deployment.

---

### 5. Model Saving

The trained model and preprocessing scaler were saved using Joblib.

- model.pkl
- scaler.pkl

This avoids retraining the model every time the application starts.

---

### 6. Streamlit Web Application

A user-friendly Streamlit interface was created that allows users to:

- Enter medical measurements
- Predict tumor diagnosis
- View prediction confidence
- Use the model directly through a web browser

---

### 7. Deployment

The complete application was deployed using **Streamlit Community Cloud**, allowing anyone to access the model online without installing Python or any dependencies.

---

# 💻 Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit

### IDE

- PyCharm

### Deployment

- Streamlit Community Cloud

### Version Control

- Git
- GitHub

---

# 📁 Project Structure

```
Breast-Cancer-Diagnosis/

│── app.py

│── requirements.txt

│── README.md

│── data/

│── models/

│── src/

└── images/
```

---

# 📸 Application Screenshots

(<img width="1889" height="906" alt="image" src="https://github.com/user-attachments/assets/094ae36b-bbf4-48d9-98ca-68bc49cd9c08" />)
(<img width="1916" height="904" alt="image" src="https://github.com/user-attachments/assets/b79f66c7-b8c7-491c-95ba-5615e99c6c0e" />)



---

# 🎯 What I Learned

This project gave me practical experience in:

- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Scaling
- Model Comparison
- Machine Learning Workflow
- Saving and Loading Models
- Streamlit Application Development
- Git & GitHub
- Cloud Deployment

More importantly, it taught me that building a Machine Learning model is only one part of the process. Delivering a complete, usable application requires proper project organization, clean code, deployment, and documentation.

---
