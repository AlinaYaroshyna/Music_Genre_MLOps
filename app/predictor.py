import os
import joblib
import numpy as np


class GenrePredictor:

    def __init__(self):
        self.models = {}
        model_dir = "artifacts/models"

        for file in os.listdir(model_dir):
            if file.endswith(".pkl"):
                model_name = file.replace(".pkl","")
                path = os.path.join(model_dir,file)
                self.models[model_name] = joblib.load(path)

    def predict(self, extracted_features):
        results = {}
        for model_name, model in self.models.items():
            if "mfcc" in model_name:
                x = [extracted_features["mfcc"]]
            elif "chroma" in model_name:
                x = [extracted_features["chroma"]]
            elif "centroid" in model_name:
                x = [extracted_features["centroid"]]
            elif "zcr" in model_name:
                x = [extracted_features["zcr"]]
            else:
                x = [extracted_features["combined"]]

            prediction = model.predict(x)[0]
            confidence = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(x)[0]
                confidence = float(np.max(probabilities))

            results[model_name] = {
                "prediction": prediction,
                "confidence": confidence
            }

        return results