from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

class ModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

    def train_model(self, model_name, X_train, y_train):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} is not available.")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        return model

    def train_all(self, X_train, y_train):
        trained_models = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
        return trained_models