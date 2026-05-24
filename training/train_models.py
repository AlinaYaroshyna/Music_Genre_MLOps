import os
import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from app.feature_extractor import FeatureExtractor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from training.save_results import save_report, save_confusion_matrix, save_metrics

def create_model(model_name):
    if model_name == "svm":
        return make_pipeline(StandardScaler(), SVC(probability=True))
    elif model_name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier())
    elif model_name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(solver="adam", max_iter=1500, tol=1e-2, random_state=42))
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":

    DATA_PATH = "data/raw"

    extractor = FeatureExtractor()

    X_data = {
        "mfcc": [],
        "chroma": [],
        "centroid": [],
        "zcr": [],
        "combined": []
    }

    y_data = []

    print("Loading dataset...")

    for genre in os.listdir(DATA_PATH):
        genre_path = os.path.join(DATA_PATH, genre)

        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)
            try:
                features = extractor.extract_all(file_path)
                if features is None:
                    continue
                for key in X_data:
                    X_data[key].append(features[key])
                y_data.append(genre)
            except Exception as e:
                print(e)

    y_data = np.array(y_data)
    for feature_name, X in X_data.items():
        X = np.array(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.2, random_state=42, stratify=y_data)
        print(f"\nTraining on {feature_name}")

        for model_name in ["svm", "knn", "mlp"]:
            os.makedirs("artifacts/models", exist_ok=True)
            model_path = f"artifacts/models/{feature_name}_{model_name}.pkl"
            model = create_model(model_name)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = balanced_accuracy_score(y_test, y_pred)

            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics = {"balanced_accuracy": accuracy}

            save_report(report,f"artifacts/reports/{feature_name}_{model_name}.json")
            save_confusion_matrix(cm,f"artifacts/confusion_matrices/{feature_name}_{model_name}.npy")
            save_metrics(metrics,f"artifacts/metrics/{feature_name}_{model_name}.json")
            print(f"{feature_name} + {model_name}: {accuracy:.4f}")

            joblib.dump(model, model_path)
            print(f"Saved: {model_path}")