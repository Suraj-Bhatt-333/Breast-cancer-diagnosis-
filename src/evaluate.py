from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

class ModelEvaluator:
    def __init__(self):
        pass

    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        
        metrics = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "F1 Score": f1_score(y_test, predictions),
            "ROC-AUC": roc_auc_score(y_test, predictions)
        }
        return metrics

    def evaluate_all(self, models_dict, X_test, y_test):
        all_results = {}
        for name, model in models_dict.items():
            all_results[name] = self.evaluate_model(model, X_test, y_test)
        return all_results